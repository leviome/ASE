import argparse
import logging
import os
import os.path as osp

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import CameraViewDataSet
from nets.ase_net import AutoSceneEncoder as ASE
from nets.utils import CycleScheduler, loss_function, decimal

def load_model(device):
    model = ASE(in_channels=4)
    model = model.to(device)
    checkpoint_dict = torch.load("checkpoints/0711/ase_1111.pt")
    model.load_state_dict(checkpoint_dict["state_dict"])
    model.eval()
    return model


def _main():
    device = "cuda:0"
    model = load_model(device)

    dataset = CameraViewDataSet(mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for img, cam, ima in dataloader:
        if ima[0] in dataset.test_list:
            print(ima[0])
            print(cam)
            img = img.to(device)
            cam = cam.to(device)

            cam2img = model.render(cam)
            utils.save_image(cam2img, "samples/novel_view/novel_{}".format(ima[0]))

            cam_code, image_code, recons, cam2img = model(cam, img)
            sample_size = 1
            sample = img[:sample_size]
            out = recons[:sample_size]
            cam_img = cam2img[:sample_size]
            merge = torch.cat([sample, out, cam_img], 0)
            utils.save_image(
                    merge,
                    "samples/novel_view/demo_{}".format(ima[0]),
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

def wild_render():
    cam = [[[ 0.3748,  0.9210, -0.1062, -0.4249],
         [-0.9271,  0.3724, -0.0429, -0.1718],
         [ 0.0000,  0.1146,  0.9934,  3.9737],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]]

    cam = [[
        [
          -0.7670794500570494,
          0.13790273552731083,
          -0.6265556263066799,
          -2.2388229992661
        ],
        [
          -0.641455362086549,
          -0.1478994690941921,
          0.7527687330728285,
          2.9308274677268766
        ],
        [
          0.011141623021434146,
          0.9793408919255631,
          0.20190909256689926,
          0.6464810204346605
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ]]


    cam = torch.Tensor(cam)
    print(cam)

    device = "cuda:0"
    model = load_model(device)
    cam = cam.to(device)

    cam2img = model.render(cam)
    utils.save_image(cam2img, "wild_cam2img.png")

def render_path():
    import json
    frames = json.loads(open("transforms.json", 'r').read())["frames"]
    cams = [f["transform_matrix"] for f in frames]
    device = "cuda:0"
    model = load_model(device)

    for i, cam in enumerate(cams):
        print(cam)
        cam = torch.Tensor([cam])
        cam = cam.to(device)
        cam2img = model.render(cam)
        utils.save_image(cam2img, "tmp/%03d.png" % i)


if __name__ == '__main__':
    # _main()
    # wild_render()
    render_path()
