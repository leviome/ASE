# WDSR_A
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
            self, n_feats, kernel_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = list()
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class SRM(nn.Module):
    """
    super resolution module
    """
    def __init__(self):
        super(SRM, self).__init__()
        # hyper-params
        scale = 4
        n_resblocks = 16
        n_feats = 64
        n_channels = 4
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        head = list()
        head.append(
            wn(nn.Conv2d(n_channels, n_feats, 3, padding=3 // 2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, 64, wn=wn, res_scale=1, act=act))

        # define tail module
        tail = []
        out_feats = scale * scale * n_channels
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2)))
        tail.append(nn.PixelShuffle(scale))

        skip = list()
        skip.append(
            wn(nn.Conv2d(n_channels, out_feats, 5, padding=5 // 2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        return x


if __name__ == '__main__':
    model = SRM()
    a = torch.randn(4, 200, 200)
    print(a.shape)
    b = model(a)
    print(b.shape)
