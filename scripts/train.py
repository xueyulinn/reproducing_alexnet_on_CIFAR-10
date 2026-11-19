import torch
import torch.nn as nn
from alexnet_cifar10 import data, engine, model


def main():
    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() else "cpu"
    print(f"current accelerator: {device}")
    num_class = 10  # class number for CIFAR-10
    batch_size = 5000
    dataloader_tr, dataloader_ts = data.load_data(batch_size)
    act = nn.ReLU()
    alexnet = model.AlexNet(act, num_class)
    alexnet.to(device)
    print(alexnet)

    # epochs = 5
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(weight_decay=5e-4)
    # for i in range(epochs):
    # engine.train(dataloader_tr, model, )


if __name__ == "__main__":
    main()
