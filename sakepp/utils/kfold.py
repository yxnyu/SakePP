import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from .training import train_and_evaluate
from ..dataset.datasets import CustomDGLDataset
import logging
from . import plot_kfold_comparison,plot_training_curves
from dgl.dataloading import GraphDataLoader
from datetime import datetime
import json


logger = logging.getLogger(f"{__name__}.kfoldDataset")


def kfold_split_dataset(dataset, n_splits=5, val_ratio=0.1, random_state=42):
    """Split dataset"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(dataset))

    fold_indices = []
    for train_val_idx, test_idx in kfold.split(indices):
        split_point = int(len(train_val_idx) * (1 - val_ratio))
        np.random.shuffle(train_val_idx)
        
        fold_indices.append({
            'train': indices[train_val_idx[:split_point]],
            'valid': indices[train_val_idx[split_point:]],
            'test': indices[test_idx],
            'sizes': {
                'train': len(indices[train_val_idx[:split_point]]),
                'valid': len(indices[train_val_idx[split_point:]]),
                'test': len(indices[test_idx])
            }
        })
    return fold_indices


def train_kfold(dataset, model_class, hyperparameters, device, n_splits=5, base_save_dir='./results_kfold', **kwargs):
    """Execute K-fold cross-validation training"""

    os.makedirs(base_save_dir, exist_ok=True)
    
    # Display optimal hyperparameters recommendation
    logger.info("=" * 60)
    logger.info("🎯 OPTIMAL HYPERPARAMETERS RECOMMENDATION")
    logger.info("=" * 60)
    logger.info("Based on extensive experiments, the following parameters")
    logger.info(f"have shown optimal performance for {n_splits}-fold cross-validation:")
    logger.info("")
    logger.info("📊 Recommended Optimal Parameters:")
    logger.info("  • batch_size: 128")
    logger.info("  • num_epochs: 80") 
    logger.info("  • learning_rate: 0.0001")
    logger.info("  • patience: 8")
    logger.info("  • dropout_rate: 0.15")
    logger.info("  • num_layers: 4")
    logger.info("  • dgn_num_layers: 8")
    logger.info("  • fusion_temperature: 0.2")
    logger.info("")
    logger.info("💡 These parameters were optimized for protein-protein")
    logger.info("   interaction prediction tasks with cross-chain modeling.")
    logger.info("=" * 60)
    logger.info("")

    # Get fold indices
    fold_indices = kfold_split_dataset(dataset, n_splits=n_splits)
    fold_results = []

    # Train for each fold
    for fold, indices in enumerate(fold_indices):
        fold_save_dir = os.path.join(base_save_dir, f'fold_{fold}')
        os.makedirs(fold_save_dir, exist_ok=True)
        
        logger.info(f'\nStarting Fold {fold + 1}/{n_splits}')
        logger.info(f'Train size: {indices["sizes"]["train"]}, '
                    f'Valid size: {indices["sizes"]["valid"]}, '
                    f'Test size: {indices["sizes"]["test"]}')

        # Create data subsets
        train_subset = Subset(dataset, indices['train'])
        valid_subset = Subset(dataset, indices['valid'])
        test_subset = Subset(dataset, indices['test'])

        train_loader = GraphDataLoader(
            train_subset,
            batch_size=hyperparameters['batch_size'],
            shuffle=True,
            num_workers=hyperparameters['num_workers'],
            drop_last=True
        )

        valid_loader = GraphDataLoader(
            valid_subset,
            batch_size=hyperparameters['batch_size'],
            shuffle=False,
            num_workers=hyperparameters['num_workers']
        )
        
        test_loader = GraphDataLoader(
            test_subset,
            batch_size=hyperparameters['batch_size'],
            shuffle=False,
            num_workers=hyperparameters['num_workers']
        )
        
        # Initialize model
        model = model_class(
            **{k: v for k, v in hyperparameters.items() 
                if k in model_class.__init__.__code__.co_varnames}).to(device)

        # Set optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Train current fold
        metric_tracker, test_metrics = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            criterion=torch.nn.MSELoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=hyperparameters['num_epochs'],
            save_dir=fold_save_dir,
            patience=hyperparameters['patience'],
            test_interval=hyperparameters['test_interval']
        )

        # Save current fold results
        fold_results.append({
            'fold': fold,
            'test_metrics': test_metrics,
            'training_history': {
                'train_losses': metric_tracker.get_all('train_loss'),
                'valid_losses': metric_tracker.get_all('valid_loss'),
                'learning_rates': metric_tracker.get_all('learning_rate')
            }
        })

        # Save current fold complete results
        torch.save(fold_results[-1], os.path.join(fold_save_dir, 'fold_results.pt'))
        with open(os.path.join(fold_save_dir, 'metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        # Plot current fold training curves
        plot_training_curves(
            metric_tracker.get_all('train_loss'),
            metric_tracker.get_all('valid_loss'),
            fold_save_dir
        )

    # Calculate and print average results
    avg_metrics = {
        'rmse': np.mean(
            [fold['test_metrics']['rmse'] for fold in fold_results]),
        'mae': np.mean(
            [fold['test_metrics']['mae'] for fold in fold_results]),
        'r2': np.mean(
            [fold['test_metrics']['r2'] for fold in fold_results]),
        'correlation': np.mean(
            [fold['test_metrics']['correlation'] for fold in fold_results])
    }

    std_metrics = {
        'rmse': np.std(
            [fold['test_metrics']['rmse'] for fold in fold_results]),
        'mae': np.std(
            [fold['test_metrics']['mae'] for fold in fold_results]),
        'r2': np.std(
            [fold['test_metrics']['r2'] for fold in fold_results]),
        'correlation': np.std(
            [fold['test_metrics']['correlation'] for fold in fold_results])
    }

    logger.info('\nK-fold Cross Validation Final Results:')
    for metric in avg_metrics:
        logger.info(
            f'Average {metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}')

    # Save complete K-fold validation results
    final_results = {
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'std_metrics': std_metrics
    }

    torch.save(
        final_results, os.path.join(
            base_save_dir, 'kfold_final_results.pt')
            )

    # Plot comparison of all folds
    plot_kfold_comparison(fold_results, base_save_dir)

    return final_results