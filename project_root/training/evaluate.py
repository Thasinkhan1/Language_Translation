import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)

            with torch.cuda.amp.autocast():
                output = model(src, trg[:, :-1])  # shift target
                loss = criterion(
                    output.reshape(-1, output.shape[-1]),
                    trg[:, 1:].reshape(-1)
                )

            total_loss += loss.item()

    return total_loss / len(loader)
