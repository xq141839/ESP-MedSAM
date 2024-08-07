import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor 
import albumentations as A

class BinaryLoader(Dataset):
        def __init__(self, data_name, jsfiles, transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'datasets'
            self.jsfiles = jsfiles
            self.img_tesnor = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ])
            self.transforms = transforms
            self.img_size = 1024
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]

            image_path = os.path.join(self.path,'image_1024/',image_id)
            mask_path = os.path.join(self.path,'mask_1024/',image_id)

            modality_name = list(image_id.split('_'))[0]

            mcls = None

            if modality_name == 'ISIC':
                mcls = 0
            elif modality_name == 'CHNCXR':
                mcls = 1
            elif modality_name == 'ultra':
                mcls = 2

            if mcls is None and len(list(image_id.split('_'))) > 1:
                mcls = 3

            if mcls is None and len(modality_name) > 10:
                mcls = 4

            if mcls is None and len(modality_name) <= 10:
                mcls = 5
 
    
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32')
            mask = io.imread(mask_path+'.png', as_gray=True)

            mask[mask>0]=255
            


            data_group = self.transforms(image=img, mask=mask)
            img_resized = data_group['image']
            mask = data_group['mask']

            img = self.img_tesnor(img)
            img = self.preprocess(img)

            # print(mask.shape)
            # mask_64 = F.interpolate(mask, scale_factor=2)
            # print(mask_64.shape)

   
            return (img_resized, img, mask, image_id, mcls)
        
        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x

