import copy
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
from .eval_vis import calculate_metrics,MetricTracker,save_checkpoint
from .val import validate,test
import logging

logger = logging.getLogger(f"{__name__}.modelTraining")


class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0


class LRSchedulerWithWarmup:
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs=10, max_lr=1e-3, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (100 - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(progress * np.pi))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


def setup_optimizer(model, learning_rate=1e-3, weight_decay=1e-5):
    """Setup optimizer"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=True
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    return optimizer, scheduler


def init_weights(model):
    """Initialize model weights"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.normal_(param, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.zeros_(param)


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'size_mb': total_params * 4 / (1024 * 1024)  # Assume each parameter occupies 4 bytes
    }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []

    for batched_graph, target in train_loader:
        # Data preparation
        batched_graph = batched_graph.to(device)
        h = batched_graph.ndata['feat'].float().to(device)
        x = batched_graph.ndata['coord'].float().to(device)
        v = torch.zeros_like(x).to(device)
        pairlist = torch.stack(batched_graph.edges(), dim=0).to(device)
        edge_attr = batched_graph.edata['attr'].float().to(device)
        target = target.to(device)

        # Forward propagation
        optimizer.zero_grad()
        output = model(batched_graph, h, x, v, pairlist, edge_attr)
        loss = criterion(output.squeeze(), target)

        # Backward propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        predictions.extend(output.detach().cpu().numpy())
        targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss, predictions, targets


def train_and_evaluate(
    model, 
    train_loader, 
    valid_loader, 
    test_loader,
    criterion, 
    optimizer, 
    scheduler, 
    device,
    num_epochs,
    save_dir,
    patience=20,
    test_interval=3
    ):
    """Complete training and evaluation process"""
    early_stopping = EarlyStopping(patience=patience)
    metric_tracker = MetricTracker()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss, val_preds, val_targets = validate(
            model, valid_loader, criterion, device)

        # Calculate metrics
        train_metrics = calculate_metrics(train_preds, train_targets)
        valid_metrics = calculate_metrics(val_preds, val_targets)

        # Update learning rate
        scheduler.step(val_loss)

        # Update metric tracker
        metric_tracker.update('train_loss', train_loss)
        metric_tracker.update('valid_loss', val_loss)
        metric_tracker.update('train_metrics', train_metrics)
        metric_tracker.update('valid_metrics', valid_metrics)
        metric_tracker.update('learning_rate', optimizer.param_groups[0]['lr'])

        # Print progress
        logger.info(f'\nEpoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')
        logger.info(f'Train RMSE: {train_metrics["rmse"]:.4f}, Valid RMSE: {valid_metrics["rmse"]:.4f}')

        # Periodic testing
        if (epoch + 1) % test_interval == 0:
            test_metrics = test(model, test_loader, criterion, device)
            logger.info(f'Test Metrics: {test_metrics}')

        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': valid_metrics
            }, is_best, save_dir)

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info('Early stopping triggered')
            break

    # Final testing
    model.load_state_dict(early_stopping.best_state)
    final_test_metrics = test(model, test_loader, criterion, device)

    return metric_tracker, final_test_metrics