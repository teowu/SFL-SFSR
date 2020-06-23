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
from network import VGG_Classifier
with open('paths.yml', 'r') as f:
    PATHS = yaml.load(f)

dataset = 'bird'
target = PATHS['bird_lr_x16']
model = VGG_Classifier().cuda()
model.load_model('../bird/prt.pth')
train_set = loader.TrainDataset(PATHS[dataset]['train'])
train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=True)

val_set = loader.TrainDataset(PATHS[dataset]['valid'], is_train=False)
val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

for image, label, indices in train_loader:
    with torch.no_grad():
        print(torch.log(torch.nn.Softmax()(model (image.cuda())) ))
        image = image.to('cuda')
        cam = model.get_cam(image, 1/16).cpu()
        mm = torch.min(cam)
        MM = torch.max(cam)
        print(mm, MM)
    save_path = os.path.join(target['cam_train'], indices[0].split('/')[-1])
    print(save_path)
    cv2.imwrite(save_path, util.tensor2img(cam, min_max=(mm,MM)))


for image, label, indices in val_loader:
    with torch.no_grad():
        image = image.to('cuda')
        cam = model.get_cam(image, 1/16).cpu()
    save_path = os.path.join(target['cam_valid'], indices[0].split('/')[-1])
    print(save_path)
    cv2.imwrite(save_path, util.tensor2img(cam))    
    