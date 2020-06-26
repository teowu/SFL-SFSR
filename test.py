#For Testing, use the test dataset
# validation
import os
import math
import argparse
import random
import logging

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options as option
import util
from data import create_dataloader, create_dataset
from clsgan_sr import CLSGAN_Model as Model
from network import PerceptualLossLPIPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    dataset_opt = opt['datasets']['test']
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt, opt, None)
    
    model = Model(opt)
    
    if test_loader is not None:  
        calc_lpips = PerceptualLossLPIPS()
        if True:
            avg_ssim = avg_psnr = avg_lpips = 0
            print("Testing Starts!")
            idx = 0
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                idx += 1
                img_name = os.path.splitext(os.path.basename(test_data['LQ_path'][0]))[0]
                img_dir = '../test_results_' + opt['name']
                util.mkdir(img_dir)

                model.feed_data(test_data)
                model.test()

                visuals = model.get_current_visuals()
                sr_img = util.tensor2img(visuals['SR'])  # uint8
                gt_img = util.tensor2img(visuals['GT'])  # uint8
                lq_img = util.tensor2img(visuals['LQ'])  # uint8

                # Save SR images for reference
                save_sr_img_path = os.path.join(img_dir,
                                                 '{:s}_sr.png'.format(img_name))

                util.save_img(sr_img, save_sr_img_path)

                gt_img = gt_img / 255.
                sr_img = sr_img / 255.
                lq_img = lq_img / 255.
                avg_psnr += util.calculate_psnr(sr_img * 255, gt_img * 255)
                avg_ssim += util.calculate_ssim(sr_img * 255, gt_img * 255)
                avg_lpips += calc_lpips(visuals['SR'], visuals['GT'])
   
                
            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx
            avg_lpips = avg_lpips / idx
                
            print('Test_Result_{:s} psnr: {:.4e} ssim: {:.4e} lpips: {:.4e}'.format(
                    opt['name'], avg_psnr, avg_ssim, avg_lpips))
                
                

if __name__ == '__main__':
    main()