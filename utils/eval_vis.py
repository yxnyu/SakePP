import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import torch


def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics, including RMSE"""
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # Calculate correlation coefficient
    correlation = np.corrcoef(predictions, targets)[0, 1]

    # Calculate MSE
    mse = np.mean((predictions - targets) ** 2)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    # Calculate MAE 
    mae = np.mean(np.abs(predictions - targets))

    # Calculate R²
    ss_total = np.sum((targets - np.mean(targets)) ** 2)
    ss_residual = np.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return {
        'correlation': correlation,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_training_curves(train_losses, valid_losses, save_dir='./plots'):
    """Plot training and validation loss curves"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    plt.figure(figsize=(12, 6))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(valid_losses, label='Valid Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot loss change rate
    plt.subplot(1, 2, 2)
    train_changes = np.diff(train_losses)
    valid_changes = np.diff(valid_losses)
    plt.plot(train_changes, label='Train Loss Change', color='blue', alpha=0.7)
    plt.plot(valid_changes, label='Valid Loss Change', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Change')
    plt.title('Loss Change Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{timestamp}.png'))
    plt.close()


def plot_kfold_comparison(fold_results, save_dir):
    """Plot K-fold cross-validation result comparison"""
    plt.figure(figsize=(15, 10))
    
    # Training loss comparison
    plt.subplot(2, 2, 1)
    for fold in fold_results:
        plt.plot(fold['training_history']['train_losses'], 
                label=f"Fold {fold['fold']+1}")
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Validation loss comparison
    plt.subplot(2, 2, 2)
    for fold in fold_results:
        plt.plot(fold['training_history']['valid_losses'], 
                label=f"Fold {fold['fold']+1}")
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Test metrics comparison
    plt.subplot(2, 2, 3)
    metrics = ['rmse', 'mae', 'r2', 'correlation']
    x = np.arange(len(metrics))
    width = 0.15

    for i, fold in enumerate(fold_results):
        values = [fold['test_metrics'][metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=f"Fold {fold['fold']+1}")

    plt.title('Test Metrics Comparison')
    plt.xticks(x + width * (len(fold_results)-1)/2, metrics)
    plt.legend()

    # Learning rate change comparison
    plt.subplot(2, 2, 4)
    for fold in fold_results:
        plt.plot(fold['training_history']['learning_rates'], 
                label=f"Fold {fold['fold']+1}")
    plt.title('Learning Rate Changes')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kfold_comparison.png'))
    plt.close()


class MetricTracker:
    """Metric tracker"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rates': [],
            'train_metrics': [],
            'valid_metrics': []
        }

    def update(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_latest(self, metric_name):
        return self.metrics[metric_name][-1] if self.metrics[metric_name] else None

    def get_all(self, metric_name):
        return self.metrics[metric_name]


def save_checkpoint(state, is_best, save_dir='./checkpoints'):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save the latest checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{timestamp}.pt')
    torch.save(state, checkpoint_path)

    # If it's the best model, save a copy
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(state, best_path)