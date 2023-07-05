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
    def __init__(self, root="/home/liweiliao/Projects/NeRF/NAE/lego", transform=transform_):
        self.transform = transform(800)
        self.root = root
        assert 'transforms_train.json' in os.listdir(root)
        with open(osp.join(root, 'transforms_train.json')) as f:
            camera_info = json.loads(f.read())
        # print(camera_info)
        self.cam = camera_info

    def __getitem__(self, i):
        pairs = self.cam["frames"][i]
        image_path = osp.join(self.root, 'train', pairs['file_path'].split('r_0')[-1])
        img = self.transform(Image.open(image_path))
        cam = torch.Tensor(np.array(pairs['transform_matrix']))
        return img, cam

    def __len__(self):
        return len(self.cam["frames"])


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
