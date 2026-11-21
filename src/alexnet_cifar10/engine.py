import torch


def train(dataloader_tr, model, loss_fn, optimizer, device):
    size = len(dataloader_tr.dataset)
    model.train()
    total_loss = 0.0
    correct_1 = 0
    for batch, (X, y) in enumerate(dataloader_tr):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * X.shape[0]
        correct_1 += (pred.argmax(1) == y).sum().item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return total_loss / size, 1 - (correct_1 / size) * 100


def test(dataloader_ts, model, loss_fn, device):
    size = len(dataloader_ts.dataset)
    model.eval()
    total_loss = 0.0
    correct_1 = 0
    with torch.no_grad():
        for X, y in dataloader_ts:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.shape[0]
            correct_1 += (pred.argmax(1) == y).sum().item()

    return total_loss / size, (1 - (correct_1 / size)) * 100
