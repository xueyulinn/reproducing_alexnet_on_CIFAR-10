import torch
from alexnet_cifar10 import data, engine, model


def main():
    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() else "cpu"
    print(device)


if __name__ == "__main__":
    main()
