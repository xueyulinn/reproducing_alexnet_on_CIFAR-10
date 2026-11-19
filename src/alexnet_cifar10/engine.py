def train(dataloader_tr, model, loss_fn, optimizer, device):
    model.train()
    # lossi = []
    for batch, (X, y) in enumerate(dataloader_tr):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"batch: {batch}, loss: {loss.item():.4f}")
