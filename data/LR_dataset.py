import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class LRDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        print(self.opt)
        self.paths_LR = None
        self.LR_env = None  # environment for lmdb
        # read image list from lmdb or image files
        opt['data_type'] = 'img'
        self.paths_LR, self.LR_env = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        assert self.paths_LR, 'Error: LR paths are empty.'

    def __getitem__(self, index):
        LR_path = None

        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)
        H, W, C = img_LR.shape

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        if self.opt['size'] is not None:
            img_LR = TF.to_pil_image(img_LR)
            img_LR = T.CenterCrop(self.opt['size'])(img_LR)
            img_LR = TF.to_tensor(img_LR)
        return {'LQ': img_LR, 'LQ_path': LR_path}

    def __len__(self):
        return len(self.paths_LR)
