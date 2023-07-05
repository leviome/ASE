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
from nets.ase_net import ASE
from nets.utils import CycleScheduler, loss_function, decimal


def _main():
    ...


if __name__ == '__main__':
    _main()
