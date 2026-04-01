import sys

import torch

from nn import mlp, cnn, Affine, residual_mlp
from para import sequential
from train import train
from visualize import create_tracker

MODEL_TYPE = "mlp"  # mlp, cnn, resmlp
DATASET = "cifar10"  # mnist, cifar10, emnist_digits, emnist_letters, emnist_balanced
EPOCHS = 25
LR = 0.0015
BATCH_SIZE = 512
WEIGHT_DECAY = 0.01
MLP_HIDDEN = [256, 128, 64]
CNN_CHANNELS = [32, 64, 128]
RESMLP_HIDDEN = 256
RESMLP_DEPTH = 3


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model_type = MODEL_TYPE
    dataset = DATASET
    epochs = EPOCHS
    lr = LR
    batch_size = BATCH_SIZE

    for arg in sys.argv[1:]:
        if arg in ("cnn", "mlp", "resmlp"):
            model_type = arg
        elif arg in ("mnist", "cifar10"):
            dataset = arg
        elif arg.startswith("emnist"):
            dataset = arg
        elif arg.startswith("epochs="):
            epochs = int(arg.split("=")[1])
        elif arg.startswith("lr="):
            lr = float(arg.split("=")[1])
        elif arg.startswith("bs="):
            batch_size = int(arg.split("=")[1])

    # Normalize emnist format: accept emnist_digits, emnist:digits, emnist-digits
    if dataset.startswith("emnist") and ":" not in dataset:
        for sep in ("_", "-"):
            if sep in dataset:
                split = dataset.split(sep, 1)[1]
                dataset = f"emnist:{split}"
                break
        else:
            if dataset == "emnist":
                dataset = "emnist:balanced"

    if dataset == "mnist":
        in_dim, in_ch, H, W, num_classes = 784, 1, 28, 28, 10
    elif dataset == "cifar10":
        in_dim, in_ch, H, W, num_classes = 3072, 3, 32, 32, 10
    elif dataset.startswith("emnist:"):
        in_dim, in_ch, H, W = 784, 1, 28, 28
        split = dataset.split(":")[1]
        num_classes = {"digits": 10, "letters": 26, "balanced": 47}[split]
    else:
        in_dim, in_ch, H, W, num_classes = 3072, 3, 32, 32, 10

    if model_type == "mlp":
        model = mlp(in_dim, MLP_HIDDEN, num_classes, act="relu")
        desc = f"MLP: {in_dim} -> {MLP_HIDDEN} -> {num_classes}"
    elif model_type == "resmlp":
        model = sequential(Affine(in_dim, RESMLP_HIDDEN), residual_mlp(RESMLP_HIDDEN, depth=RESMLP_DEPTH), Affine(RESMLP_HIDDEN, num_classes))
        desc = f"ResMLP: {in_dim} -> {RESMLP_HIDDEN} ({RESMLP_DEPTH} res blocks) -> {num_classes}"
    else:
        model = cnn(in_ch, H, W, CNN_CHANNELS, num_classes, act="relu")
        desc = f"CNN: {in_ch}ch -> {CNN_CHANNELS} -> {num_classes}"

    print(f"Model: {desc}")
    print(f"Dataset: {dataset}\n")

    tracker = create_tracker(save_dir=f"results/catdl/{model_type}/{dataset}")
    train(model, epochs=epochs, lr=lr, batch_size=batch_size, dataset=dataset, device=device, tracker=tracker, model_name=model_type, weight_decay=WEIGHT_DECAY)


if __name__ == "__main__":
    main()
