import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import lr_scheduler
from network import *
import logging
import os

logger = logging.getLogger('base')

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True, subpath=None):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        if subpath != None:
            load_net =  load_net[subpath]
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
            

class CLSGAN_Model(BaseModel):
    def __init__(self, opt):
        super(CLSGAN_Model, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        G_opt = opt['network_G']

        # define networks and load pretrained models
        self.netG = RCAN(G_opt).to(self.device)
        self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = Discriminator_VGG_256(3, G_opt['nf']).to(self.device)
            self.netD = DataParallel(self.netD)
            self.netG.train()
            self.netD.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = VGGFeatureExtractor(feature_layer=34, use_bn=False,
                                          use_input_norm=True, device=self.device).to(self.device)
                self.netF = DataParallel(self.netF)
                    
            # G feature loss
            if train_opt['cls_weight'] > 0:
                l_cls_type = train_opt['cls_criterion']
                if l_cls_type == 'CE':
                    self.cri_cls = nn.NLLLoss().to(self.device)
                elif l_cls_type == 'l1':
                    self.cri_cls = nn.L1Loss().to(self.device)
                elif l_cls_type == 'l2':
                    self.cri_cls = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_cls_type))
                self.l_cls_w = train_opt['cls_weight']
            else:
                logger.info('Remove classification loss.')
                self.cri_cls = None
            if self.cri_cls:  # load VGG perceptual loss
                self.netC = VGGFeatureExtractor(feature_layer=49, use_bn=True,
                                          use_input_norm=True, device=self.device).to(self.device)
                load_path_C = self.opt['path']['pretrain_model_C']
                assert load_path_C is not None, "Must get Pretrained Classfication prior."
                self.netC.load_model(load_path_C)  
                self.netC = DataParallel(self.netC)
                
            if train_opt['brc_weight'] > 0:
                self.l_brc_w = train_opt['brc_weight'] 
                self.netR = VGG_Classifier().to(self.device)
                load_path_C = self.opt['path']['pretrain_model_C']
                assert load_path_C is not None, "Must get Pretrained Classfication prior."
                self.netR.load_model(load_path_C)  
                self.netR = DataParallel(self.netR)


            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H, self.cls_L = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
                
            if self.cri_cls: # F-G classification loss
                #print(self.netC(self.var_H).detach().shape)
                #real_cls = self.netC(self.var_H).argmax(1).detach()
                #fake_cls = torch.log( nn.Softmax(dim=1) (self.netC(self.fake_H)) )
                real_cls = self.netC(self.var_H).detach()
                fake_cls = self.netC(self.fake_H)
                l_g_cls = self.l_cls_w * self.cri_cls(fake_cls, real_cls)
                l_g_total = l_g_cls
            if self.opt['train']['gan_type'] == 'gan':
                pred_g_fake = self.netD(self.fake_H)
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                pred_g_fake = self.netD(self.fake_H)
                l_g_gan = self.l_gan_w * (
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan
            
            if self.opt['train']['br_optimizer'] == 'joint':
                ref = self.netR(self.var_H).argmax(dim=1)
                l_branch = self.l_brc_w * nn.CrossEntropyLoss()(self.cls_L, ref)
                self.optimizer_G.step()

            l_g_total.backward()
            self.optimizer_G.step()
        
            self.optimizer_G.zero_grad()
            
            # seperate branching update
            if self.opt['train']['br_optimizer'] == 'branch':
                ref = self.netR(self.var_H).argmax(dim=1)
                l_branch = self.l_brc_w * nn.CrossEntropyLoss()(self.cls_L, ref)
                self.optimizer_G.step()
                
        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        if self.opt['train']['gan_type'] == 'gan':
            # need to forward and backward separately, since batch norm statistics differ
            # real
            pred_d_real = self.netD(self.var_ref)
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_fake.backward()
        elif self.opt['train']['gan_type'] == 'ragan':
            # pred_d_real = self.netD(self.var_ref)
            # pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            # l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            # l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            # l_d_total = (l_d_real + l_d_fake) / 2
            # l_d_total.backward()
            pred_d_fake = self.netD(self.fake_H.detach()).detach()
            pred_d_real = self.netD(self.var_ref)
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True) * 0.5
            l_d_real.backward()
            pred_d_fake = self.netD(self.fake_H.detach())
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real.detach()), False) * 0.5
            l_d_fake.backward()
        self.optimizer_D.step()
        
        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, _ = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)
                    
            if self.cri_cls:  # C, F-G Classification Network
                s, n = self.get_network_description(self.netC)
                if isinstance(self.netC, nn.DataParallel) or isinstance(
                        self.netC, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netC.__class__.__name__,
                                                     self.netC.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netC.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network C structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):  
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
    def clear_data(self):
        return None