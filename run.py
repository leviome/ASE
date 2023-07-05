from nets import AutoEncoder
from dataset.dataset import *
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim


def _main():
    vae = AutoEncoder(in_channels=4).cuda()
    dataset = NerfData()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    loss_func = MSELoss(reduction='mean')

    optimizer = optim.SGD(vae.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-6)
    for i in range(10000):
        for batch in dataloader:
            batch = batch.cuda()
            optimizer.zero_grad()
            z = vae.encode(batch)
            result = vae.decode(z)
            loss = loss_func(batch, result)
            print(loss)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    _main()
