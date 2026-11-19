from torchvision import datasets
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader


def load_data(batch_size=5000):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    data_tr = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    data_ts = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    # 50000 10000
    print(len(data_tr), len(data_ts))
    dataloader_tr = DataLoader(data_tr, batch_size=batch_size)
    dataloader_ts = DataLoader(data_ts, batch_size=batch_size)
    return dataloader_tr, dataloader_ts
