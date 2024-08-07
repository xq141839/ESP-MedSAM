# ESP-MedSAM: Efficient Self-Prompting SAM for Universal Domain-Generalized Medical Image Segmentation

:pushpin: This is an official PyTorch implementation of **ESP-MedSAM: Efficient Self-Prompting SAM for Universal Domain-Generalized Medical Image Segmentation**

[[`arXiv`]()] [[`BibTeX`]()]


<div align="center">
    <img width="100%" alt="UKAN overview" src="framework.png"/>
</div>

## 📰News

**[2024.08.07]** Paper will be updated soon!

**[2024.08.07]** Code and model checkpoints are released!

## 🛠Setup

```bash
git clone https://github.com/xq141839/ESP-MedSAM.git
cd ESP-MedSAM
conda create -n ESP python=3.10
conda activate ESP
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install albumentations==0.5.2
pip install pytorch_lightning==1.1.1
pip install monai
```

**Note**: Please refer to requirements.txt


## 📚Data Preparation

The structure is as follows.
```
ESP-MedSAM
├── datasets
│   ├── image_1024
│     ├── ISIC_0000000.png
|     ├── ...
|   ├── mask_1024
│     ├── ISIC_0000000.png
|     ├── ...
```

## 🎪Segmentation Model Zoo
We provide all pre-trained models here.
| MA-Backbone | MC | Checkpoints |
|-----|------|-----|
|TinyViT| Dermoscopy | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=0SHI0m)|
|TinyViT| X-ray | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=0SHI0m)|
|TinyViT| Fundus | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=0SHI0m)|
|TinyViT| Colonoscopy | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=0SHI0m)|
|TinyViT| Ultrasound | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=0SHI0m)|
|TinyViT| Microscopy | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=0SHI0m)|


## 🎈Acknowledgements
Greatly appreciate the tremendous effort for the following projects!
- [SAM](https://github.com/facebookresearch/segment-anything)
- [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT)

