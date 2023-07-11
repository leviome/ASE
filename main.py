import argparse
import logging
import os
import os.path as osp

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from torch.nn import SyncBatchNorm

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from dataset.dataset import CameraViewDataSet
from nets.ase_net import AutoSceneEncoder as ASE
from nets.utils import CycleScheduler, loss_function, decimal, psnr

log_writer = SummaryWriter()
DATE = '0711_1'


class Trainer:
    def __init__(self, dataset, args, world_size=4, gpu_id=0, logger=None, multi_process=False):
        self.gpu_id = gpu_id
        self.logger = logger
        self.lr = args.lr
        self.epochs = args.epochs
        self.start_epoch = 0
        self.mp = multi_process
        self.batch_size = args.batch_size // world_size
        self.date = None
        model = ASE(in_channels=4)
        model = model.to(self.gpu_id)

        self.model = DDP(model, device_ids=[self.gpu_id]) if self.mp else model.to(self.gpu_id)
        if args.resume and osp.exists(args.checkpoint):
            logger.info(f"resuming and loading checkpoint {args.checkpoint}.")
            checkpoint_dict = torch.load(args.checkpoint)
            if "state_dict" in checkpoint_dict.keys():
                self.model.load_state_dict(checkpoint_dict["state_dict"])
                self.start_epoch = checkpoint_dict["epoch"]
            else:
                self.model.load_state_dict(checkpoint_dict)
        if self.mp:
            self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                         pin_memory=True, shuffle=False,
                                         sampler=DistributedSampler(dataset))
        else:
            self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CycleScheduler(
            self.optimizer,
            self.lr,
            n_iter=len(self.dataloader) * self.epochs,
            momentum=None,
            warmup_proportion=0.05,
        )

    def train_one_epoch(self, epoch_i):
        self.model.train()
        if self.mp:
            self.dataloader.sampler.set_epoch(epoch_i)
        for i, (img, cam) in enumerate(self.dataloader):
            self.model.zero_grad()
            img = img.to(self.gpu_id)
            cam = cam.to(self.gpu_id)
            cam_code, image_code, recons, cam2img = self.model(cam, img)
            self_recon_loss, cam_bind_loss, cam2img_loss = loss_function(cam_code, image_code,
                                                                         recons, img,
                                                                         cam2img, bind_mode='mse')
            loss = self_recon_loss + cam_bind_loss + cam2img_loss
            loss.backward()

            self.scheduler.step()
            self.optimizer.step()

            if self.gpu_id == 0 and epoch_i % 5 == 0 and i == 17:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"epoch:{epoch_i}\t"
                      f"loss:{decimal(loss.item())}\tlr:{decimal(lr)}\n"
                      f"self_recon:{decimal(self_recon_loss.item())}\t"
                      f"cam_bind:{decimal(cam_bind_loss.item())}\t"
                      f"cam2img:{decimal(cam2img_loss.item())}")

                log_writer.add_scalar('Loss/overall', decimal(loss.item()), epoch_i)
                log_writer.add_scalar('Loss/self_recon', decimal(self_recon_loss.item()), epoch_i)
                log_writer.add_scalar('Loss/cam_bind', decimal(cam_bind_loss.item()), epoch_i)
                log_writer.add_scalar('Loss/cam2img', decimal(cam2img_loss.item()), epoch_i)

                sample_size = 4
                sample = img[:sample_size]
                out = recons[:sample_size]
                cam_img = cam2img[:sample_size]
                merge = torch.cat([sample, out, cam_img], 0)
                utils.save_image(
                    merge,
                    f"samples/{self.date}/{str(epoch_i).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

    def train(self, date):
        self.date = date
        if not osp.exists(f"checkpoints/{date}"):
            os.system(f'mkdir -p checkpoints/{date}')
        if not osp.exists(f"samples/{date}"):
            os.system(f'mkdir -p samples/{date}')

        self.model.train()
        for i in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_one_epoch(i)
            if self.gpu_id == 0 and i % 10 == 0 and i > 1:
                save_dict = dict()
                save_dict["state_dict"] = self.model.module.state_dict() if self.mp else self.model.state_dict()
                save_dict["epoch"] = i
                save_dict["date"] = date
                save_dict["description"] = ""
                torch.save(save_dict, f"checkpoints/{date}/ase_{str(i + 1).zfill(3)}.pt")


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # nccl is NVIDIA Collective Communications Library, a backend used for distributed communications across CUDA GPUs.

    torch.manual_seed(42)


def run_inference(model, dataset):
    model.eval()
    with torch.no_grad():
        ...


def single_process(rank: int, world_size: int, dataset, args, date, logger):
    # logger.info(f"starting process [{rank}].")
    print(f"starting process [{rank}].")
    ddp_setup(rank, world_size)

    trainer = Trainer(dataset, args, gpu_id=rank, world_size=world_size,
                      logger=logger, multi_process=True)
    trainer.train(date)

    destroy_process_group()


def mp_main():
    date = DATE
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(asctime)s: %(message)s',
                        # filename=f'{date}_train.log',
                        # filemode='w'
                        )
    logger = logging.getLogger(f"{date}_log")
    dataset = CameraViewDataSet()

    world_size = torch.cuda.device_count()
    mp.spawn(single_process, args=(world_size, dataset, args, date, logger), nprocs=world_size)
    # rank will be auto-assigned by mp.spawn


def get_args():
    parser = argparse.ArgumentParser("Clay", add_help=False)
    parser.add_argument('--lr', default=0.003)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--resume', default=False)
    parser.add_argument('--checkpoint', default="./checkpoints/0522/ase_911.pt", type=str)
    parser.add_argument('--epochs', default=100000, type=int)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--local_rank', type=str, help='local rank for dist')

    return parser.parse_args()


def _main():
    date = DATE
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(asctime)s: %(message)s',
                        # filename=f'{date}_train.log',
                        # filemode='w'
                        )
    logger = logging.getLogger(f"{date}_log")
    dataset = CameraViewDataSet()

    trainer = Trainer(dataset, args, logger=logger)
    trainer.train(date)

    return


if __name__ == '__main__':
    # _main()
    mp_main()
