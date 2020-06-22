import torch
import time
import os
import yaml
import data_loader as loader
from tqdm import tqdm
import torch.nn.functional as F
import cv2

with open('paths.yml', 'r') as f:
    PATHS = yaml.load(f)

dataset = 'bird'
target = PATHS['bird_lr']

train_set = loader.TrainDataset(PATHS[dataset]['train'])
train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=True)

val_set = loader.TrainDataset(PATHS[dataset]['valid'], is_train=False)
val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)


for image, label, indices in tqdm(train_loader):
    with torch.no_grad():
        image = image.to('cuda')
        lr = F.interpolate(x, scale_factor=1/8, mode='bicubic', align_corners=False).cpu()
    save_path = os.path.join(target['train'], indices.split('/')[-1])
    print(save_path)
    time.sleep(20)
    cv2.imwrite(lr, save_path)


for image, label, indices in tqdm(val_loader):
    with torch.no_grad():
        image = image.to('cuda')
        lr = F.interpolate(x, scale_factor=1/8, mode='bicubic', align_corners=False).cpu()
    save_path = os.path.join(target['val'], indices.split('/')[-1])
    print(save_path)
    time.sleep(20)
    cv2.imwrite(lr, save_path)    
    