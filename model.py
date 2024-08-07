import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from SAM.modeling.mask_decoder import MaskDecoder
from SAM.modeling.prompt_encoder import PromptEncoder
from SAM.modeling.transformer import TwoWayTransformer
from SAM.modeling.common import LayerNorm2d
from typing import List, Tuple, Type, Optional
from SAM.modeling.student_encoder import SemiTViT
import os
from tqdm import tqdm
from functools import partial
from timm.models import create_model


class ESPMedSAM(nn.Module):
    def __init__(self):
        super(ESPMedSAM, self).__init__()

        self.SemiTViT = SemiTViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )

        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64), # 1024 // 16
            input_image_size=(1024, 1024),
            mask_in_chans=16,
            )
        
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            )
        
        self.patch_decoder =  nn.Sequential(nn.Conv2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1))
        
        self.dense_prompter = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1),
            LayerNorm2d(256),
            nn.GELU())

        self.num_mask_tokens = 7
        self.mask_query = nn.Embedding(self.num_mask_tokens, 256)


  
    def forward(self, x, mask=None, domain_seq=1, img_id=None):

        b = x.shape[0]

        image_embeddings = self.SemiTViT(x)

        patch_map = self.patch_decoder(image_embeddings)
        prob_map = torch.sigmoid(patch_map.detach())
        prob_map[prob_map >= 0.5] = 1
        prob_map[prob_map < 0.5] = 0
        prob_map = F.interpolate(prob_map, scale_factor=2)
        patch_embeddings = self.dense_prompter(prob_map)
        

        outputs_mask = []

        for idx in range(b): # for each batch 

            image_embeddings_dec = image_embeddings[idx].unsqueeze(0)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

            image_pe = self.prompt_encoder.get_dense_pe()

            # Mask 
            mask_tokens = self.mask_query.weight
            mask_tokens = mask_tokens.unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
            tokens_mask = torch.cat((mask_tokens, sparse_embeddings), dim=1) # 1 x 5 x 256
            # Expand per-image data in batch direction to be per-mask
            mask_src = torch.repeat_interleave(image_embeddings_dec, tokens_mask.shape[0], dim=0)
            mask_src = mask_src + patch_embeddings[idx].unsqueeze(0) # 1 x 256 x 64 x 64
            mask_pos_src = torch.repeat_interleave(image_pe, tokens_mask.shape[0], dim=0)  # 1 x 256 x 64 x 64

            low_res_masks = self.mask_decoder(
                src=mask_src,
                pos_src=mask_pos_src,
                tokens=tokens_mask,
                mcls=domain_seq#[idx]
            )

            masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)

            outputs_mask.append(masks.squeeze(0))


        return torch.stack(outputs_mask, dim=0), patch_map, image_embeddings
        