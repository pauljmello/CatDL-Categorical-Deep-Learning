import csv
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self, save_dir="results"):
        self.save_dir = Path(str(save_dir).replace(":", "_"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.test_accs: List[float] = []
        self.epoch_times: List[float] = []
        self.csv_path: Optional[Path] = None

    def init_csv(self, model_name, dataset, lr, batch_size):
        timestamp = Path(f"{model_name}_lr{lr}_bs{batch_size}")
        self.csv_path = self.save_dir / f"{timestamp}.csv"
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_accuracy', 'epoch_time_seconds'])
        return self.csv_path

    def log_epoch(self, epoch, train_loss, test_acc, epoch_time=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_accs.append(test_acc)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

        if self.csv_path is not None:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if epoch_time is not None:
                    writer.writerow([epoch, train_loss, test_acc, epoch_time])
                else:
                    writer.writerow([epoch, train_loss, test_acc, ''])

    def plot_metrics(self, model_name, dataset, save=True):
        if not self.epochs:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'{model_name} on {dataset}', fontsize=14, fontweight='bold')

        ax1.plot(self.epochs, self.train_losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.epochs, self.test_accs, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Test Accuracy', fontsize=12)
        ax2.set_title('Test Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()

        if save:
            plot_path = self.save_dir / f"{model_name}_metrics.png"
            plt.savefig(plot_path, bbox_inches='tight')
            print(f"\nPlot saved to: {plot_path}")

        return fig


    def save_summary(self, model_name, dataset):
        """
        Save a summary of the training run.
        """
        if not self.epochs:
            return

        summary_path = self.save_dir / f"{model_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary: {model_name} on {dataset}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Epochs: {len(self.epochs)}\n")
            f.write(f"Final Training Loss: {self.train_losses[-1]:.4f}\n")
            f.write(f"Final Test Accuracy: {self.test_accs[-1]:.4f}\n")
            f.write(f"Best Test Accuracy: {max(self.test_accs):.4f} (Epoch {self.epochs[self.test_accs.index(max(self.test_accs))]})\n")
            f.write(f"Lowest Training Loss: {min(self.train_losses):.4f} (Epoch {self.epochs[self.train_losses.index(min(self.train_losses))]})\n")

            if self.epoch_times:
                total_time = sum(self.epoch_times)
                avg_time = total_time / len(self.epoch_times)
                f.write(f"\nTiming Information:\n")
                f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)\n")
                f.write(f"Average Time per Epoch: {avg_time:.2f} seconds\n")
                f.write(f"Fastest Epoch: {min(self.epoch_times):.2f} seconds (Epoch {self.epochs[self.epoch_times.index(min(self.epoch_times))]})\n")
                f.write(f"Slowest Epoch: {max(self.epoch_times):.2f} seconds (Epoch {self.epochs[self.epoch_times.index(max(self.epoch_times))]})\n")

        print(f"Summary saved to: {summary_path}")

        return summary_path


def create_tracker(save_dir="results"):
    return MetricsTracker(save_dir=save_dir)
