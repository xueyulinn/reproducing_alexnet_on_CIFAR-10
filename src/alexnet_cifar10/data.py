from torchvision import datasets
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def load_data(batch_size=128, aug=True):
    ''' 
    Load the CIFAR-10 dataset, which contains 60000 32x32x3-images 
    with 50000 training images and 10000 training images. There are 
    10 classes in total, it takes 4 bits to map the image to label
    and 0.2 milliom constraints in total, and the number of trainable parameters 
    is around 7 million.  
    '''
    aug = [v2.RandomCrop(32, padding=4),
           v2.RandomHorizontalFlip()] if aug else []
    base = [v2.ToImage(), v2.ToDtype(torch.float32),
            v2.Normalize(mean=MEAN, std=STD)]
    tf_tr = v2.Compose(aug + base)
    tf_ts = v2.Compose(base)
    data_tr = datasets.CIFAR10(
        root="data", train=True, download=True, transform=tf_tr
    )
    data_ts = datasets.CIFAR10(
        root="data", train=False, download=True, transform=tf_ts
    )

    dataloader_tr = DataLoader(data_tr, batch_size=batch_size)
    dataloader_ts = DataLoader(data_ts, batch_size=batch_size)
    return dataloader_tr, dataloader_ts
