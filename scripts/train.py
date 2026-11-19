import torch
import torch.nn as nn
from alexnet_cifar10 import data, engine, model


def main():
    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() else "cpu"
    print(f"current accelerator: {device}")

    # load data
    num_class = 10  # class number for CIFAR-10
    batch_size = 128  # batch size in paper
    dataloader_tr, dataloader_ts = data.load_data(batch_size)

    # load model
    act = nn.ReLU()
    alexnet = model.AlexNet(act, num_class)
    alexnet.to(device)
    n_trainable = sum(p.numel()
                      for p in alexnet.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_trainable}")

    # training
    epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(alexnet.parameters(), weight_decay=5e-4)
    for i in range(epochs):
        print(f"epoch{i}: ------------------------------")
        engine.train(dataloader_tr, alexnet, loss_fn, optimizer, device)
        engine.test(dataloader_ts, alexnet, loss_fn, device)
        if i > 1:
            break


if __name__ == "__main__":
    main()
