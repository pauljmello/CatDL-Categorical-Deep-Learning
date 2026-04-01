import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import torch
import torch.nn as nn

from train import mnist_loaders, emnist_loaders, cifar10_loaders


class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class BaselineCNN(nn.Module):
    def __init__(self, in_ch, H, W, channels, num_classes):
        super().__init__()
        blocks = []
        curr_ch = in_ch
        for ch in channels:
            blocks.append(nn.Conv2d(curr_ch, ch, 3, padding=1))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2))
            curr_ch = ch

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(curr_ch, num_classes)
        self.in_ch = in_ch
        self.H = H
        self.W = W

    def forward(self, x):
        x = x.view(x.size(0), self.in_ch, self.H, self.W)
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.fc2(torch.tanh(self.fc1(x)))


class BaselineResMLP(nn.Module):
    def __init__(self, in_dim, hidden, depth, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.input_proj(x.view(x.size(0), -1))
        x = self.blocks(x)
        return self.output_proj(x)


DATASET_INFO = {
    "mnist": (784, 1, 28, 28, 10),
    "cifar10": (3072, 3, 32, 32, 10),
    "emnist:digits": (784, 1, 28, 28, 10),
    "emnist:letters": (784, 1, 28, 28, 26),
    "emnist:balanced": (784, 1, 28, 28, 47),
}


def dataset_info(dataset):
    return DATASET_INFO[dataset]


def get_loaders(dataset, batch_size):
    if dataset == "mnist":
        return mnist_loaders(batch_size)
    elif dataset == "cifar10":
        return cifar10_loaders(batch_size)
    else:
        return emnist_loaders(batch_size, split=dataset.split(":")[1])


def build_model(model_type, dataset, mlp_hidden, cnn_channels, resmlp_hidden, resmlp_depth):
    in_dim, in_ch, H, W, num_classes = dataset_info(dataset)
    if model_type == "mlp":
        return BaselineMLP(in_dim, mlp_hidden, num_classes)
    elif model_type == "cnn":
        return BaselineCNN(in_ch, H, W, cnn_channels, num_classes)
    else:
        return BaselineResMLP(in_dim, resmlp_hidden, resmlp_depth, num_classes)


def train_baseline(model_type, dataset, epochs, lr, batch_size, device, weight_decay, mlp_hidden, cnn_channels, resmlp_hidden, resmlp_depth):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_type, dataset, mlp_hidden, cnn_channels, resmlp_hidden, resmlp_depth).to(device)
    train_loader, test_loader = get_loaders(dataset, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    final_loss = 0.0
    history = []
    start_time = time.time()

    print(f"[PyTorch] Training {model_type} on {dataset} for {epochs} epochs (lr = {lr})")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(dim=-1) == yb).sum().item()
                total += yb.numel()

        acc = correct / max(total, 1)
        epoch_time = time.time() - epoch_start

        if acc > best_acc:
            best_acc = acc

        final_loss = avg_loss
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, "time": epoch_time})
        print(f"  Epoch {epoch:2d}: loss = {avg_loss:.4f}, test_acc = {acc:.4f}, time = {epoch_time:.2f}s")

    total_time = time.time() - start_time
    return {"best_accuracy": best_acc, "final_loss": final_loss, "total_time_s": total_time, "history": history}
