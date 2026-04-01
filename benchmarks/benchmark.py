import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from nn import mlp, cnn, residual_mlp, Affine
from para import sequential
from train import train_epoch, eval_accuracy, SoftmaxCrossEntropy, AdamW
from baseline import train_baseline, dataset_info, get_loaders

MODELS = ["mlp", "cnn", "resmlp"]
DATASETS = ["mnist", "emnist:digits", "emnist:letters", "emnist:balanced", "cifar10"]
EPOCHS = 25
LR = 0.0015
BATCH_SIZE = 512
WEIGHT_DECAY = 0.01
MLP_HIDDEN = [256, 128, 64]
CNN_CHANNELS = [32, 64, 128]
RESMLP_HIDDEN = 256
RESMLP_DEPTH = 3
MASTER_CSV = Path(__file__).resolve().parent.parent / "results" / "benchmark" / "master.csv"


def run_catdl(model_type, dataset, device):
    in_dim, in_ch, H, W, num_classes = dataset_info(dataset)

    if model_type == "mlp":
        model = mlp(in_dim, MLP_HIDDEN, num_classes, act="relu")
    elif model_type == "cnn":
        model = cnn(in_ch, H, W, CNN_CHANNELS, num_classes, act="relu")
    else:
        model = sequential(Affine(in_dim, RESMLP_HIDDEN), residual_mlp(RESMLP_HIDDEN, depth=RESMLP_DEPTH), Affine(RESMLP_HIDDEN, num_classes))

    train_loader, test_loader = get_loaders(dataset, BATCH_SIZE)
    params = model.init(device=device)
    opt = AdamW(weight_decay=WEIGHT_DECAY)
    opt_state = opt.init_state(params)
    loss_fn = SoftmaxCrossEntropy()

    best_acc = 0.0
    final_loss = 0.0
    history = []
    start_time = time.time()

    print(f"[CatDL] Training {model_type} on {dataset} for {EPOCHS} epochs (lr = {LR})")
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        params, avg_loss, opt_state = train_epoch(model, params, train_loader, opt, opt_state, LR, loss_fn, device)
        acc = eval_accuracy(model, params, test_loader, device)
        epoch_time = time.time() - epoch_start

        if acc > best_acc:
            best_acc = acc

        final_loss = avg_loss
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, "time": epoch_time})
        print(f"  Epoch {epoch:2d}: loss = {avg_loss:.4f}, test_acc = {acc:.4f}, time = {epoch_time:.2f}s")

    total_time = time.time() - start_time
    return {"best_accuracy": best_acc, "final_loss": final_loss, "total_time_s": total_time, "history": history}


def plot_axis(ax, epochs, catdl_vals, pytorch_vals, ylabel, ylim=None):
    ax.plot(epochs, catdl_vals, "b-o", linewidth=2, markersize=5, label="CatDL")
    ax.plot(epochs, pytorch_vals, "r-s", linewidth=2, markersize=5, label="PyTorch")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)


def save_comparison(model_type, dataset, catdl, pytorch):
    safe_dataset = dataset.replace(":", "_")
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "results" / "benchmark" / model_type / safe_dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    catdl_hist = catdl["history"]
    pytorch_hist = pytorch["history"]

    csv_path = out_dir / f"{model_type}_lr{LR}_bs{BATCH_SIZE}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "catdl_train_loss", "catdl_test_accuracy", "catdl_epoch_time", "pytorch_train_loss", "pytorch_test_accuracy", "pytorch_epoch_time"])
        for c, p in zip(catdl_hist, pytorch_hist):
            writer.writerow([c["epoch"], f"{c['loss']:.4f}", f"{c['acc']:.4f}", f"{c['time']:.2f}", f"{p['loss']:.4f}", f"{p['acc']:.4f}", f"{p['time']:.2f}"])
    print(f"  CSV saved to: {csv_path}")

    epochs = []
    catdl_losses = []
    pytorch_losses = []
    catdl_accs = []
    pytorch_accs = []

    for h in catdl_hist:
        epochs.append(h["epoch"])
        catdl_losses.append(h["loss"])
        catdl_accs.append(h["acc"])

    for h in pytorch_hist:
        pytorch_losses.append(h["loss"])
        pytorch_accs.append(h["acc"])

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_type.upper()} on {dataset} -- CatDL vs PyTorch", fontsize=14, fontweight="bold")
    plot_axis(ax_loss, epochs, catdl_losses, pytorch_losses, "Training Loss")
    plot_axis(ax_acc, epochs, catdl_accs, pytorch_accs, "Test Accuracy", ylim=[0, 1])
    plt.tight_layout()

    plot_path = out_dir / f"{model_type}_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_path}")

    summary_path = out_dir / f"{model_type}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Benchmark: {model_type.upper()} on {dataset}\n"
                f"{'=' * 50}\n\n"
                f"{'Metric':<25} {'CatDL':>12} {'PyTorch':>12}\n"
                f"{'-' * 49}\n"
                f"{'Best Test Accuracy':<25} {catdl['best_accuracy']:>11.4f}  {pytorch['best_accuracy']:>11.4f}\n"
                f"{'Final Training Loss':<25} {catdl['final_loss']:>11.4f}  {pytorch['final_loss']:>11.4f}\n"
                f"{'Total Time (s)':<25} {catdl['total_time_s']:>11.2f}  {pytorch['total_time_s']:>11.2f}\n")
    print(f"  Summary saved to: {summary_path}")


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiments: {len(MODELS)} models x {len(DATASETS)} datasets x 2 frameworks")

    existing = set()
    if MASTER_CSV.exists():
        with open(MASTER_CSV, "r") as f:
            for row in csv.DictReader(f):
                existing.add((row["framework"], row["model"], row["dataset"]))

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not MASTER_CSV.exists():
        with open(MASTER_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["framework", "model", "dataset", "best_accuracy", "final_loss", "total_time_s", "epochs", "lr", "batch_size"])

    for model_type in MODELS:

        for dataset in DATASETS:
            if ("catdl", model_type, dataset) in existing and ("pytorch", model_type, dataset) in existing:
                print(f"\nSkipping {model_type}/{dataset} (already in master.csv)")
                continue

            print(f"\n{model_type.upper()} / {dataset}")

            catdl_result = run_catdl(model_type, dataset, device)
            print()

            pytorch_result = train_baseline(model_type, dataset, EPOCHS, LR, BATCH_SIZE, device, WEIGHT_DECAY, MLP_HIDDEN, CNN_CHANNELS, RESMLP_HIDDEN, RESMLP_DEPTH)
            print()

            save_comparison(model_type, dataset, catdl_result, pytorch_result)

            for framework, result in [("catdl", catdl_result), ("pytorch", pytorch_result)]:
                with open(MASTER_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([framework, model_type, dataset, f"{result['best_accuracy']:.4f}", f"{result['final_loss']:.4f}", f"{result['total_time_s']:.2f}", EPOCHS, LR, BATCH_SIZE])

            print(f"\n  CatDL:   best_acc = {catdl_result['best_accuracy']:.4f}, loss = {catdl_result['final_loss']:.4f}, time = {catdl_result['total_time_s']:.1f}s")
            print(f"  PyTorch: best_acc = {pytorch_result['best_accuracy']:.4f}, loss = {pytorch_result['final_loss']:.4f}, time = {pytorch_result['total_time_s']:.1f}s")

    print(f"\nAll results saved to {MASTER_CSV}")


if __name__ == "__main__":
    run_benchmark()
