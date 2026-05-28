import argparse
import os.path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_sessions


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        if hidden_size > 0:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.net = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

def main(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "training_data")

    X, y = load_sessions(data_dir)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features, "
          f"{y.sum()} positive / {(y == 0).sum()} negative")

    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * (1.0 - args.test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=args.batch_size)

    torch.manual_seed(args.model_seed)
    model     = Classifier(input_size=X.shape[1], hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_test_loss = float("inf")
    best_test_acc  = 0.0
    best_epoch     = 0
    best_state     = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_acc  = test_acc
            best_epoch     = epoch
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | "
                  f"test loss {test_loss:.4f} | test acc {test_acc:.3f}")

        if args.patience and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"\nBest model (epoch {best_epoch}) | test loss {best_test_loss:.4f} | test acc {best_test_acc:.3f}")

    if args.output:
        torch.save(best_state, args.output)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--test-size",  type=float, default=0.2,  help="fraction of data for test set")
    parser.add_argument("--split-seed", type=int,   default=42,   help="seed for train/test split")
    parser.add_argument("--model-seed", type=int,   default=0,    help="seed for model weight init")
    parser.add_argument("--log-every",  type=int,   default=10,   help="print metrics every N epochs")
    parser.add_argument("--patience",   type=int,   default=None, help="early stopping patience (epochs); disabled if not set")
    parser.add_argument("--hidden-size", type=int,  default=32,   help="size of hidden layer")
    parser.add_argument("--output",      type=str,  default=None, help="path to save best model weights (.pt)")
    main(parser.parse_args())
