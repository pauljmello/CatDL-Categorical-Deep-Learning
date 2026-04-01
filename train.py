import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

DATA_DIR = str(Path(__file__).resolve().parent / "data")

from euc import tree_map, tree_map2, tree_map3
from lens import training_step


class SoftmaxCrossEntropy:
    """
    Mean cross-entropy for logits with optional label smoothing.
    """

    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = float(label_smoothing)

    def loss_and_grad(self, logits, labels):
        C = int(logits.shape[-1])
        z = logits.reshape(-1, C)
        y = labels.reshape(-1)
        B = int(y.numel())

        logZ = torch.logsumexp(z, dim=1, keepdim=True)
        log_probs = z - logZ
        probs = torch.exp(log_probs)

        ls = self.label_smoothing
        if ls == 0.0:
            loss = -log_probs[torch.arange(B, device=y.device), y].mean()
            d = probs.clone()
            d[torch.arange(B, device=y.device), y] -= 1.0
            d = d / B
            return loss, d.view_as(logits)

        with torch.no_grad():

            target = torch.full((B, C), ls / C, device=z.device, dtype=z.dtype)
            target[torch.arange(B, device=y.device), y] += 1.0 - ls

        loss = -(target * log_probs).sum(dim=1).mean()
        d = (probs - target) / B
        return loss, d.view_as(logits)


@dataclass
class AdamW:
    """
    AdamW: Adam with decoupled weight decay.
    """
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    def init_state(self, params):
        m = tree_map(torch.zeros_like, params)
        v = tree_map(torch.zeros_like, params)
        return (0, m, v)

    @torch.no_grad()
    def step(self, params, state, grads, lr):
        t, m, v = state
        t1 = int(t) + 1
        b1 = float(self.beta1)
        b2 = float(self.beta2)
        eps = float(self.eps)
        wd = float(self.weight_decay)
        lr = float(lr)

        def update_m(g, m0):
            return b1 * m0 + (1.0 - b1) * g

        def update_v(g, v0):
            return b2 * v0 + (1.0 - b2) * (g * g)

        m1 = tree_map2(update_m, grads, m)
        v1 = tree_map2(update_v, grads, v)

        b1t = 1.0 - (b1 ** t1)
        b2t = 1.0 - (b2 ** t1)

        def upd_p(p, m_new, v_new):
            if wd != 0.0:
                p_wd = p * (1.0 - lr * wd)
            else:
                p_wd = p
            return p_wd - lr * ((m_new / b1t) / (torch.sqrt(v_new / b2t) + eps))

        return tree_map3(upd_p, params, m1, v1), (t1, m1, v1)


def flatten(x):
    return x.view(-1)


def shift_label(y):
    return y - 1


LOADER_KWARGS = dict(num_workers=4, persistent_workers=True, prefetch_factor=2)


def mnist_loaders(batch_size=128):
    from torchvision import transforms, datasets

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten)
    ])

    train = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    return (DataLoader(train, batch_size=batch_size, shuffle=True, **LOADER_KWARGS), DataLoader(test, batch_size=batch_size, shuffle=False, **LOADER_KWARGS))


def emnist_loaders(batch_size=128, split='balanced'):
    from torchvision import transforms, datasets

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten)
    ])

    target_transform = shift_label if split == 'letters' else None

    train = datasets.EMNIST(root=DATA_DIR, split=split, train=True, download=True, transform=transform, target_transform=target_transform)
    test = datasets.EMNIST(root=DATA_DIR, split=split, train=False, download=True, transform=transform, target_transform=target_transform)

    return (DataLoader(train, batch_size=batch_size, shuffle=True, **LOADER_KWARGS), DataLoader(test, batch_size=batch_size, shuffle=False, **LOADER_KWARGS))


def cifar10_loaders(batch_size=128):
    from torchvision import transforms, datasets

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(flatten),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(flatten),
    ])

    train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    return (DataLoader(train, batch_size=batch_size, shuffle=True, **LOADER_KWARGS), DataLoader(test, batch_size=batch_size, shuffle=False, **LOADER_KWARGS))


def train_epoch(model, params, loader, opt, opt_state, lr, loss_fn=None, device=None):
    """
    One epoch of training.
    Returns (new_params, avg_loss, new_opt_state).
    """
    if loss_fn is None:
        loss_fn = SoftmaxCrossEntropy()

    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        if device is not None:
            xb, yb = xb.to(device), yb.to(device)

        if xb.dim() > 2:
            xb = xb.view(xb.size(0), -1)

        params, loss, opt_state = training_step(model, loss_fn, opt, xb, yb, params, opt_state, lr)

        total_loss += float(loss.detach().item())
        n_batches += 1

    return params, total_loss / max(n_batches, 1), opt_state



@torch.no_grad()
def eval_accuracy(model, params, loader, device=None):
    """
    Evaluate accuracy.
    """
    correct = 0
    total = 0

    for xb, yb in loader:
        if device is not None:
            xb, yb = xb.to(device), yb.to(device)

        if xb.dim() > 2:
            xb = xb.view(xb.size(0), -1)

        logits = model(params, xb)
        preds = logits.argmax(dim=-1)
        correct += int((preds == yb).sum().item())
        total += int(yb.numel())

    return correct / max(total, 1)



def train(model, epochs=5, lr=1e-3, batch_size=128, dataset="mnist", device=None, tracker=None, model_name="model", weight_decay=0.01):
    """
    Full training run with logging.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if dataset == "mnist":
        train_loader, test_loader = mnist_loaders(batch_size)
    elif dataset == "cifar10":
        train_loader, test_loader = cifar10_loaders(batch_size)
    elif dataset.startswith("emnist"):
        if ":" in dataset:
            split = dataset.split(":")[1]
        else:
            split = "balanced"
        train_loader, test_loader = emnist_loaders(batch_size, split=split)
    else:
        train_loader, test_loader = cifar10_loaders(batch_size)

    params = model.init(device=device)
    opt = AdamW(weight_decay=weight_decay)
    opt_state = opt.init_state(params)
    loss_fn = SoftmaxCrossEntropy()

    if tracker is not None:
        tracker.init_csv(model_name, dataset, lr, batch_size)

    print(f"Training for {epochs} epochs on {dataset} (lr = {lr})")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        params, avg_loss, opt_state = train_epoch(model, params, train_loader, opt, opt_state, lr, loss_fn, device)
        acc = eval_accuracy(model, params, test_loader, device)
        epoch_time = time.time() - epoch_start_time

        print(f"  Epoch {epoch:2d}: loss = {avg_loss:.4f}, test_acc = {acc:.4f}, time = {epoch_time:.2f}s")

        if tracker is not None:
            tracker.log_epoch(epoch, avg_loss, acc, epoch_time)

    if tracker is not None:

        tracker.plot_metrics(model_name, dataset, save=True)
        tracker.save_summary(model_name, dataset)

    return params
