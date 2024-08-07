import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import ESPMedSAM
# from MLSPSAM import MEMedSAM
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric


# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_num_threads(4)

def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y)
    tmp_hd = torch.mean(tmp_hd)

    return tmp_hd.item()

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='isic',type=str, help='isic, cvccdb, lung, retinal, dsb, ultrasound')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--model',default='ESP_isic_best.pth', type=str, help='the path of model')
    args = parser.parse_args()

    pred_path = f'visual/pred/{args.dataset}/'

    os.makedirs(pred_path, exist_ok=True)

    args.jsonfile = f'{args.dataset}_data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test'] #+ df['train'] + df['valid']
    test_dataset = BinaryLoader(args.dataset, test_files, A.Compose([
                                        A.Resize(1024, 1024),
                                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensor()
                                        ]))

    model = ESPMedSAM()


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(args.model), strict=True)

    model = model.cuda()

    model.train(False)  
    
    TestAcc = Accuracy()
    TestPrecision = Precision()
    TestDice = Dice()
    TestRecall = Recall()
    TestF1 = F1(2)
    TestIoU = IoU()

    mIoU = []
    Accuracy = []
    Precision = []
    Recall = []
    F1_score = []
    DSC = []
    FPS = []
    image_ids = []
    hd_list = []
    
    since = time.time()
    
    with torch.no_grad():
        for _, img, mask, img_id, mcls in tqdm(test_dataset):

            img = Variable(torch.unsqueeze(img, dim=0), requires_grad=False).cuda()            
            mask = Variable(torch.unsqueeze(mask, dim=0), requires_grad=False).cuda()

            torch.cuda.synchronize()
            start = time.time()

            mask_pred, _, lrmask = model(x=img, domain_seq=mcls)

            torch.cuda.synchronize()
            end = time.time()
            FPS.append(end-start)

            mask_pred = torch.sigmoid(mask_pred)

            mask_pred[mask_pred >= 0.5] = 1
            mask_pred[mask_pred < 0.5] = 0


            mask_draw = mask_pred.clone().detach()
            gt_draw = mask.clone().detach()
            

            IoU = TestIoU(mask_pred,mask)
            dsc = TestDice(mask_pred,mask)
            hdscore = hd_score(mask_pred,mask)

            mask_pred = mask_pred.view(-1)
            mask = mask.view(-1)


            img_id = list(img_id.split('.'))[0]
            mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
            mask_numpy[mask_numpy==1] = 255 
            cv2.imwrite(f'{pred_path}{img_id}.png',mask_numpy)

            gt_numpy = gt_draw.cpu().detach().numpy()[0][0]
            gt_numpy[gt_numpy==1] = 255 
            cv2.imwrite(f'{pred_path}gt_{img_id}.png',gt_numpy)

     
            accuracy = TestAcc(mask_pred.cpu(),mask.cpu())
            precision = TestPrecision(mask_pred.cpu(),mask.cpu())
            recall = TestRecall(mask_pred.cpu(),mask.cpu())
            f1score = TestF1(mask_pred.cpu(),mask.cpu())

         
            mIoU.append(IoU.item())
            DSC.append(dsc.item())
            Accuracy.append(accuracy.item())
            Precision.append(precision.item())
            Recall.append(recall.item())
            F1_score.append(f1score.item())
            image_ids.append(img_id)
            if hdscore != float("inf") and np.isnan(hdscore) != True:
                hd_list.append(hdscore)
            torch.cuda.empty_cache()

    time_elapsed = time.time() - since

    result_dict = {'image_id':image_ids, 'miou':mIoU, 'dice':DSC}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f'{pred_path}results_{args.dataset}.csv',index=False)
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    F1 = 2 * np.mean(Precision) * np.mean(Recall) / (np.mean(Precision) + np.mean(Recall))
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(FPS)/len(FPS))))
    print('mean IoU:',round(np.mean(mIoU),4),round(np.std(mIoU),4))
    print('mean accuracy:',round(np.mean(Accuracy),4),round(np.std(Accuracy),4))
    print('mean F1:',round(np.mean(F1),4))
    print('mean HD:',round(np.mean(hd_list),4),round(np.std(hd_list),4))
    print('mean Dice:',round(np.mean(DSC),4),round(np.std(DSC),4))

