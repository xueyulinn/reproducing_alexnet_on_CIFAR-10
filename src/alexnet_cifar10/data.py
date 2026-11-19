from torchvision import datasets
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader


def load_data(batch_size=128):
    ''' 
    Load the CIFAR-10 dataset, which contains 60000 32x32x3-images 
    with 50000 training images and 10000 training images. There are 
    10 classes in total, it takes 4 bits to map the image to label
    and 0.2 milliom constraints in total, and the number of trainable parameters 
    is around 7 million.  
    '''
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    data_tr = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    data_ts = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    dataloader_tr = DataLoader(data_tr, batch_size=batch_size)
    dataloader_ts = DataLoader(data_ts, batch_size=batch_size)
    return dataloader_tr, dataloader_ts
