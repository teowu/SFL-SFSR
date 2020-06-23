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
from srgan import SRGAN_Model as Baseline_Model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            print(opt['path'])
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key and path is not None))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            trial = 0
            while os.path.isdir('../Loggers/' + opt['name'] + '/' + str(trial)):
                trial += 1
            tb_logger = SummaryWriter(log_dir='../Loggers/' + opt['name'] + '/' + str(trial))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # -------------------------------------------- ADDED --------------------------------------------
    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        l1_loss = l1_loss.cuda()
        mse_loss = mse_loss.cuda()
    # -----------------------------------------------------------------------------------------------

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = Model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        train_bar = tqdm(train_loader, desc='[%d/%d]' % (epoch, total_epochs))
        for bus, train_data in enumerate(train_bar):

             # validation
            if epoch % opt['train']['val_freq'] == 0 and bus == 0 and rank <= 0 and epoch != 0:
                avg_ssim = avg_psnr = avg_psnr_n = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0
                print("into validation!")
                idx = 0
                val_bar = tqdm(val_loader, desc='[%d/%d]' % (epoch, total_epochs))
                for val_data in val_bar:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8
                    lq_img = util.tensor2img(visuals['LQ'])  # uint8
                    #nr_img = util.tensor2img(visuals['NR'])  # uint8
                    #nf_img = util.tensor2img(visuals['NF'])  # uint8
                    #nh_img = util.tensor2img(visuals['NH'])  # uint8


                    #print("Great! images got into here.")

                    # Save SR images for reference
                    save_sr_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}_sr.png'.format(img_name, current_step))
                    save_nr_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}_lq.png'.format(img_name, current_step))
                    #save_nf_img_path = os.path.join(img_dir,
                                                # 'bs_{:s}_{:d}_nr.png'.format(img_name, current_step)) 
                    #save_nh_img_path = os.path.join(img_dir,
                                                # 'bs_{:s}_{:d}_nh.png'.format(img_name, current_step)) 
                    util.save_img(sr_img, save_sr_img_path)
                    util.save_img(lq_img, save_nr_img_path)
                    #util.save_img(nf_img, save_nf_img_path)
                    #util.save_img(nh_img, save_nh_img_path)


                    #print("Saved")
                    # calculate PSNR
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    #nf_img = nf_img / 255.
                    lq_img = lq_img / 255.
                    #cropped_lq_img = lq_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    #cropped_nr_img = nr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(sr_img * 255, gt_img * 255)
                    avg_ssim += util.calculate_ssim(sr_img * 255, gt_img * 255)
                    #avg_psnr_n += util.calculate_psnr(cropped_lq_img * 255, cropped_nr_img * 255)

                    # ----------------------------------------- ADDED -----------------------------------------
                    val_pix_err_nf += l1_loss(visuals['SR'], visuals['GT'])
                    val_mean_color_err += mse_loss(visuals['SR'].mean(2).mean(1), visuals['GT'].mean(2).mean(1))
                    # -----------------------------------------------------------------------------------------
                
                
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                #avg_psnr_n = avg_psnr_n / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx



                # log
                logger.info('# Validation # PSNR: {:.4e}, {:.4e},'.format(avg_psnr, avg_psnr_n))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('val_psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('val_ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('val_pix_err_nf', val_pix_err_nf, current_step)
                    tb_logger.add_scalar('val_mean_color_err', val_mean_color_err, current_step)

            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.clear_data()
            #### tb_logger
            if current_step % opt['logger']['tb_freq'] == 0:
                logs = model.get_current_log()
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    for k, v in logs.items():
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)

            
            #### logger
            if epoch % opt['logger']['print_freq'] == 0  and epoch != 0 and bus == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                if rank <= 0:
                    logger.info(message)

           
            #### save models and training states
            if epoch % opt['logger']['save_checkpoint_freq'] == 0 and epoch != 0 and bus == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
