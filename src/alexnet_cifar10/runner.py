import torch
import torch.nn as nn
import json
from alexnet_cifar10 import data, engine, model

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
}

device = torch.accelerator.current_accelerator().type \
    if torch.accelerator.is_available() else "cpu"
print(f"current accelerator: {device}")


def run(name, seed, results_dir, aug=True, act="relu", dropout=0.5, epochs=2):
    torch.manual_seed(seed)
    # load data
    batch_size = 128  # batch size in paper
    dataloader_tr, dataloader_ts = data.load_data(batch_size, aug)

    # load model
    num_class = 10  # class number for CIFAR-10
    alexnet = model.AlexNet(ACTIVATIONS[act], dropout, num_class)
    alexnet.to(device)
    # n_trainable = sum(p.numel()
    # for p in alexnet.parameters() if p.requires_grad)
    # print(f"trainable parameters: {n_trainable}")

    # training & testing
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(alexnet.parameters(), weight_decay=5e-4)
    history = {"train_loss": [], "test_loss": [],
               "top1_error": [], "top5_error": []}
    for _ in range(epochs):
        train_loss = engine.train(
            dataloader_tr, alexnet, loss_fn, optimizer, device)
        test_loss, top1_error, top5_error = engine.test(
            dataloader_ts, alexnet, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["top1_error"].append(top1_error)
        history["top5_error"].append(top5_error)

    with open(results_dir / f"{name}_seed{seed}.json", "w") as f:
        json.dump(history, f)
