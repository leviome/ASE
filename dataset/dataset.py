import os
import os.path as osp
import numpy as np
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


def transform_(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=BICUBIC),
        ToTensor(),
    ])


class NerfData(Dataset):
    def __init__(self, transform=transform_, dealing_size=256):
        dataset_path = '/home/liweiliao/Projects/NeRF/nerf-pytorch/data/nerf_synthetic/lego/train/'
        path_list = os.listdir(dataset_path)
        self.path_list = [os.path.join(dataset_path, p) for p in path_list]
        self.transform = transform(dealing_size)
        self.sample_list = list()

    def __getitem__(self, index):
        img = Image.open(self.path_list[index])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.path_list)


class CameraViewDataSet(Dataset):
    def __init__(self, root="/home/liweiliao/Projects/NeRF/NAE/lego", transform=transform_, mode='train'):
        assert mode in ["train", "test"]
        self.mode = mode
        self.transform = transform(800)
        self.root = root
        assert 'transforms_train.json' in os.listdir(root)
        with open(osp.join(root, 'transforms_train.json')) as f:
            camera_info = json.loads(f.read())
        # print(camera_info)
        self.cam = camera_info
        self.frames = self.cam["frames"]
        # self.test_list = ["0371.png", "0424.png", "0688.png", "0795.png"]
        separate_point = (9 * len(self.frames)) // 10
        self.train_samples = self.cam["frames"][:separate_point]
        self.test_samples = self.cam["frames"][separate_point:]
        

    def __getitem__(self, i):
        samples = self.train_samples if self.mode == 'train' else self.test_samples
        pairs = samples[i]
        ima = pairs['file_path'].split('r_0')[-1]
        image_path = osp.join(self.root, 'train', ima)
        img = self.transform(Image.open(image_path))
        cam = torch.Tensor(np.array(pairs['transform_matrix']))
        return img, cam

    def __len__(self):
        return len(self.train_samples) if self.mode == 'train' else len(self.test_samples)


def _test():
    ds = NerfData()
    img = ds[0]
    print(img.shape)


def _test1():
    ds = CameraViewDataSet()
    print(ds.__len__())
    img, cam = ds[0]
    print(img.shape)
    print(cam)


if __name__ == "__main__":
    _test1()
