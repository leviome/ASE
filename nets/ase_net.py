from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F

from nets.base import BaseNet
from nets.types_ import *
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1,
                 out_padding=1, deconv=False):
        super(ResBlock, self).__init__()
        self.is_skip_connection = (in_channels == out_channels) and stride != 2
        if not deconv:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU())
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=out_padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU())

    def forward(self, x):
        # shortcut = x
        out = self.net(x)
        # if self.is_skip_connection:
        #     out += shortcut
        return out


def build_image_encoder(in_channels: int = 3):
    """
    800x800xC -> 25x25x1
    """
    hidden_dims = [32, 256, 1024, 256, 1]
    strides = [2, 2, 2, 2, 2]
    modules = []
    for h_dim, stride in zip(hidden_dims, strides):
        block = ResBlock(in_channels, h_dim, stride=stride)
        modules.append(block)
        in_channels = h_dim

    return nn.Sequential(*modules)


def build_mixer():
    mixer = TransformerEncoder(TransformerEncoderLayer(d_model=625, nhead=25, batch_first=True),
                               num_layers=6)

    return mixer


def build_image_decoder(out_channels: int = 3):
    """
    25x25x16 -> 100x100x(64xC) -> 800x800xC
    """

    # hidden_dims = [1, 128, 256, 128, 32]
    # hidden_dims = [16, 256, 1024, 256, 32]
    hidden_dims = [16, 256, 1024, 256, 256]
    strides = [1, 1, 1, 2, 2]
    modules = []
    for i in range(len(hidden_dims) - 1):
        block = ResBlock(hidden_dims[i], hidden_dims[i + 1],
                         stride=strides[i],
                         out_padding=1 if strides[i] == 2 else 0,
                         deconv=True)
        modules.append(block)

    block = ResBlock(hidden_dims[-1], hidden_dims[-1], stride=strides[-1], deconv=True)
    modules.append(nn.Sequential(
        block,
        nn.Conv2d(hidden_dims[-1], out_channels=out_channels * 64,
                  kernel_size=3, padding=1),
        nn.Tanh()))

    pixel_shuffle = nn.PixelShuffle(8)
    modules.append(pixel_shuffle)

    return nn.Sequential(*modules)


def build_cam_transform():
    """
    16 -> 625 = (25x25)
    """
    return nn.Sequential(
        nn.Linear(16, 784),
        nn.ReLU(),
        nn.Linear(784, 625)
    )


def build_content_lib_net():
    """
    625 = (25x25) -> (25x25x16)
    """
    net = nn.Sequential(
        nn.Linear(625, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4 * 4096),
        nn.ReLU(),
        nn.Linear(4 * 4096, 25 * 25 * 16),
    )
    return net


class AutoSceneEncoder(BaseNet, ABC):
    def __init__(self,
                 in_channels: int = 3) -> None:
        super(AutoSceneEncoder, self).__init__()
        self.out_channels = in_channels

        self.image_encoder = build_image_encoder(in_channels)
        self.image_decoder = build_image_decoder(self.out_channels)
        self.cam_encoder = build_cam_transform()
        # self.mixer = build_mixer()
        self.CLN = build_content_lib_net()
        self.flatten = nn.Flatten()

    def encode(self, image: Tensor) -> Tensor:
        """
        800x800xC -> 25x25x1
        :return: (Tensor) List of latent codes
        """
        return self.image_encoder(image)

    def decode(self, features: Tensor) -> Tensor:
        """
        25x25x16 -> 800x800xC
        """
        return self.image_decoder(features)

    def restore(self, image: Tensor) -> Tensor:
        z = self.image_encoder(image)
        image_code = self.flatten(z)
        features_image = self.CLN(image_code)
        restored_image = self.image_decoder(features_image.view(-1, 16, 25, 25))
        return restored_image

    def cam_encode(self, cam: Tensor) -> Tensor:
        return self.cam_encoder(cam)

    def forward(self, cam, x):
        cam = self.flatten(cam)  # 4x4 -> 16
        cam_code = self.cam_encode(cam)  # 16 -> 625
        features_cam = self.CLN(cam_code)  # 625 -> (25x25x16)
        # features_cam = features_cam.view(-1, 15, 625)
        # features_cam = torch.cat((features_cam, cam_code.unsqueeze(1)), 1)  # 15x625 -> 16x625
        # features_cam = self.mixer(features_cam)
        features_cam = features_cam.view(-1, 16, 25, 25)

        z = self.image_encoder(x)
        image_code = self.flatten(z)
        features_image = self.CLN(image_code)
        # features_image = features_image.view(-1, 15, 625)
        # features_image = torch.cat((features_image, image_code.unsqueeze(1)), 1)  # 15x625 -> 16x625
        # features_image = self.mixer(features_image)
        features_image = features_image.view(-1, 16, 25, 25)
        recons = self.image_decoder(features_image)

        cam2img = self.image_decoder(features_cam)

        # return features_cam, features_image, recons, cam2img
        return cam_code, image_code, recons, cam2img

    def render(self, cam):
        cam = self.flatten(cam)  # 4x4 -> 16
        cam_code = self.cam_encode(cam)  # 16 -> 625
        features_cam = self.CLN(cam_code)  # 625 -> (25x25x15)
        features_cam = features_cam.view(-1, 16, 25, 25)
        cam2img = self.image_decoder(features_cam)
        return cam2img

    def loss_function(self, cam, x):
        cam = self.flatten(cam)
        z = self.image_encoder(x)
        image_code = self.flatten(z)
        features_image = self.CLN(image_code)
        recons = self.image_decoder(features_image.view(-1, 16, 25, 25))

        recons_loss = F.mse_loss(recons, x)
        cam_code = self.cam_encode(cam)

        cam_bind_loss = F.l1_loss(image_code, cam_code)

        return recons_loss, cam_bind_loss

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples


def _test():
    from dataset.dataset import CameraViewDataSet
    cvd = CameraViewDataSet()
    img, cam = cvd[0]
    img = img.unsqueeze(0)
    cam = cam.unsqueeze(0)

    model = AutoSceneEncoder(in_channels=4)
    # print(model(cam, img))
    recon_loss, cam_bind_loss = model.loss_function(cam, img)
    print(recon_loss)
    print(cam_bind_loss)


if __name__ == '__main__':
    _test()
