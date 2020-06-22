from os import listdir, walk
from os.path import join
from PIL import Image
import csv
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random



class TrainDataset(Dataset):
    def __init__(self, path_dir, is_train=True, target_dir=None, grayscale=False, flips=False, rotations=False, is_test=False, **kwargs):
        super(TrainDataset, self).__init__()
        print(flips, rotations)
        self.filesX = []
        if is_test:
            walker = walk(path_dir)
            for parent, _, filenames in walker:
                for filename in filenames:
                    self.filesX.append(join(parent, filename))
            self.is_test = True
            return
        else:
            self.is_test = False

        if is_train:
            self.is_train = True
            self.labels = join(path_dir, 'train.csv')
        else:
            self.is_train = False
            self.labels = join(path_dir, 'valid.csv')
        #print(self.labels)
        with open(self.labels, 'r') as label:
            self.labels = list(csv.reader(label))
        

        for img, label in self.labels:
            if img.split('.')[-1] != 'jpg':
                img += '.jpg'
            img_path = join(path_dir, img)
            #print(img_path,)
            self.filesX.append((img_path, label))

            #print(img_path)


        # intitialize image transformations and variables
        self.f_transform = T.Compose([
            T.RandomVerticalFlip(0.1),
            T.RandomHorizontalFlip(0.1)
        ])
        self.g_transform = T.RandomGrayscale(0.01)
        self.rotations = rotations
        self.grayscale = grayscale
        self.flips = flips

    def __getitem__(self, index):
        
        if self.is_test:
            image = Image.open(self.filesX[index])
            return TF.to_tensor(image), self.filesX[index]
        image = Image.open(self.filesX[index][0])
        label = int(self.filesX[index][1])
        if self.rotations and self.is_train:
            angle = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 180, 270])
            image = TF.rotate(image, angle)
        if self.grayscale and self.is_train:
            image = self.g_transform(image)
        if self.flips and self.is_train:
            image = self.f_transform(image)
        image = TF.to_tensor(image)
        return image, label, self.filesX[index][0]

    def __len__(self):
        return len(self.filesX)
