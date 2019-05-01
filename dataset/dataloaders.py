import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class GenericVideo(Dataset):
    def __init__(self, root_dir, train=True, frame_length=10, frame_stride=1, frame_dilation=0, dimension=3, iext='tif', transforms=None, video_transforms=None):
        self.train = train
        self.root_dir = root_dir
        self.prefix = 'Train' if self.train else 'Test'
        self.dir = os.path.join(self.root_dir, self.prefix)
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.frame_dilation = frame_dilation
        self.dimension = dimension
        self.iext = iext
        self.transforms = transforms
        self.video_transforms = video_transforms
        self.list = self._getlist()
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        folder_name, indices = self.list[index]
        imgs = []
        for idx in indices:
            # img = Image.open(os.path.join(self.dir, '{}{:03d}'.format(self.prefix, folder_name + 1), '{:03d}.{}'.format(idx + 1, self.iext)))
            img = Image.open(idx)
            if self.transforms is not None:
                img = self.transforms(img)

            imgs.append(img)
        if self.video_transforms is not None:
            imgs = self.video_transforms(imgs)

        if self.dimension == 3:
            return imgs, folder_name
        else:
            return imgs.view((-1,) + imgs.shape[-2:]), folder_name
        # return torch.stack(imgs, dim=1) if self.dimension == 3 else torch.cat(imgs, dim=0), folder_name

    def _getlist(self):
        lists = []
        for folder in os.listdir(self.dir):
            images = np.array(sorted(glob.glob(os.path.join(self.dir, folder, '*.{}'.format(self.iext)))))
            max_images = len(images)
            for j in self._getstartindices(max_images):
                lists.append((folder, images[self._getindices(j)]))
        return lists
    
    def _getstartindices(self, max_images):
        return list(range(0, max_images - self._getdilatedlength() + 1, self.frame_stride))

    def _getdilatedlength(self):
        return self.frame_length * (self.frame_dilation + 1) - self.frame_dilation

    def _getindices(self, start_id):
        return list(range(start_id, start_id + self._getdilatedlength(), self.frame_dilation + 1))