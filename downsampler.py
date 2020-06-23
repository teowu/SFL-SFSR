import torch
import time
import os
import yaml
import cls_loader as loader
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import cv2
import util
with open('paths.yml', 'r') as f:
    PATHS = yaml.load(f)

dataset = 'bird'
target = PATHS['bird_lr_x16']

train_set = loader.TrainDataset(PATHS[dataset]['train'])
train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=True)

val_set = loader.TrainDataset(PATHS[dataset]['valid'], is_train=False)
val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)


for image, label, indices in train_loader:
    with torch.no_grad():
        image = image.to('cuda')
        lr = F.interpolate(image, scale_factor=1/8, mode='bilinear', align_corners=False).cpu()
    save_path = os.path.join(target['train'], indices[0].split('/')[-1])
    print(save_path)
    cv2.imwrite(save_path, util.tensor2img(lr))


for image, label, indices in val_loader:
    with torch.no_grad():
        image = image.to('cuda')
        lr = F.interpolate(image, scale_factor=1/8, mode='bilinear', align_corners=False).cpu()
    save_path = os.path.join(target['valid'], indices[0].split('/')[-1])
    print(save_path)
    cv2.imwrite(save_path, util.tensor2img(lr))    
    