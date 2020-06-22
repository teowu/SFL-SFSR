## This trainer is for the stage one classification model. We aim to achieve SR and hard classification with no direct class knowledge, so a pretrained HR classification model is to be constructed.

import sys
import time
import math
from network import VGG_Classifier
import cls_loader as loader
import time
import torch.nn as nn

import numpy as np
import torch
import os, shutil
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm
from collections import OrderedDict

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = True
        
    parser = argparse.ArgumentParser(description='Training Reference Classifier')
    parser.add_argument('--dataset', default='bird', type=str, help='selecting datasets')
    parser.add_argument('--labels_count', default=180, type=int, help='total labels count')
    parser.add_argument('--cri', default='CE', type=str, help='selecting loss functions, CE=Cross Entropy')
    parser.add_argument('--flips', default=True, type=bool, help='Data Augmentation: Flip')
    parser.add_argument('--rotations', default=True, type=bool, help='Data Augmentation: Rotation')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size used')
    parser.add_argument('--device_ids', default=(0,1,2,3), type=tuple, help='number of devices used')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used')
    parser.add_argument('--num_epochs', default=31, type=int, help='total train epoch number') 
    parser.add_argument('--save_path', default='bird', type=str, help='additional folder for saving the data')
    parser.add_argument('--val_interval', default=1, type=int, help='validation interval')
    parser.add_argument('--save_freq', default=3, type=int, help='save model frequency')
    parser.add_argument('--wei_pen', default=True, type=bool, help='save model frequency')
    parser.add_argument('--reweight', default=True, type=bool, help='save model frequency')
    parser.add_argument('--pretrained', default=True, type=bool, help='save model frequency')

    opt = parser.parse_args()
        
    with open('paths.yml', 'r') as stream:
        PATHS = yaml.load(stream)
        
    train_set = loader.TrainDataset(PATHS[opt.dataset]['train'], **vars(opt))
    val_set = loader.TrainDataset(PATHS[opt.dataset]['valid'], **vars(opt), is_train=False) 

    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers,
                              batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1)
    
    model = VGG_Classifier(labcnt=opt.labels_count, pretrained=opt.pretrained)
    
    if opt.save_path is None:
        save_path = ''
    else:
        save_path = '/' + opt.save_path
    dir_index = 0
    while os.path.isdir( 'stage_1_loggers' + save_path + '/' + str(dir_index)):
        dir_index += 1
    summary_path = 'stage_1_loggers' + save_path + '/' + str(dir_index)
    writer = SummaryWriter(summary_path)
    print('Saving summary into directory ' + summary_path + '/')
    
    
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    optimizer = torch.optim.Adam(optim_params, lr=opt.learning_rate, betas=[0.9, 0.999])
    
    iteration = 0
    total_loss = 0
    total_correctness = 0
    
    model = model.cuda()

    for epoch in range(opt.num_epochs):
        if epoch % opt.val_interval == 0:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False    
            idx = 0
            positive = 0
            T5_positive = 0
            negative = 0
            loss = 0
            val_accuracy_dict = OrderedDict()
            val_total_dict = {}
            for image, label, indices in tqdm(val_loader, desc='Validating in epoch {}'.format(epoch)):
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda() 
                y = model(image)
                index = torch.argmax(y[0])
                #print(y, index, label)
                loss += F.cross_entropy(y, label).detach()
                if index == label:
                    positive += 1
                    flag = 1
                else: negative += 1; flag = 0
                label = int(label.cpu())
                indexes = []
                for i in range(5):
                    tmp = int(torch.argmax(y[0]).cpu())
                    y[0][tmp] = -20.0
                    indexes.append(tmp)
                if label in indexes:
                    T5_positive += 1
                idx += 1

            for ele in val_accuracy_dict.keys():
                val_accuracy_dict[ele] /= val_total_dict[ele]
            
            print("###Validation accuracy in epoch {:d} is {:.2f}%, come on!####".format(epoch, positive/idx*100))
            writer.add_scalars('Validation/Classified Accuracy', val_accuracy_dict, epoch)
            writer.add_scalar('Validation/Cross Entropy Loss', loss / idx, epoch)
            writer.add_scalar('Validation/Error Rate', negative / idx, epoch) 
            writer.add_scalar('Validation/Top-5 Error Rate', 1 - T5_positive / idx, epoch) 
            model.train()
        
        # where to save the model
        if epoch % opt.save_freq == 0:
            path = opt.save_path + '/{:d}_lr={:e}.pth'.format(opt.pretrained, opt.learning_rate)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            torch.save(model.state_dict(), path)
        
        for param in model.parameters():
            param.requires_grad = True
        
        for image, label, indices in tqdm(train_loader, desc='In epoch {}'.format(epoch)):
            #print(label, indices)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            
            y = model(image)
            for i in range(int(y.shape[0])):
                index = torch.argmax(y[i])
                if index == label[i]:
                    total_correctness += 1

            loss = F.cross_entropy(y, label)
            total_loss += loss.detach()
            loss.backward()

            optimizer.step()

            k = 0

            iteration += 1
            
            if iteration % 50 == 0:
                total_loss /= 50
                total_correctness /= 1600
                writer.add_scalar('Train/Cross Entropy Loss', total_loss, iteration)
                writer.add_scalar('Train/Correctness', total_correctness, iteration)
                writer.add_scalar
                total_correctness = 0
                total_loss = 0
            torch.cuda.empty_cache()
    