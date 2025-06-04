import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
from sakepp import setup_logging, train_kfold, CustomDGLDataset
from sakepp.models import DGNLinearFusion, SAKEPP
import json
import numpy as np
import os
from datetime import datetime


os.environ["HYDRA_FULL_ERROR"] = "1"


def get_save_path(cfg) -> Path:
    
    save_path = Path(cfg.paths.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def get_device(cfg: DictConfig) -> torch.device:
    """Automatically select training device"""
    device_str = cfg.experiment.device
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def setup_seed(seed: int):
    """Set all random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_model_class(cfg: DictConfig):
    model_class = cfg.models.get("model_class", "DGNLinearFusion")  
    
    if model_class == "DGNLinearFusion":
        return DGNLinearFusion
    elif model_class == "SAKEPP":
        return SAKEPP
    else:
        raise ValueError(f"Unknown model type: {model_class}. "
                         f"Current supported: 'dgn_linear_fusion'")


@hydra.main(version_base="1.1", config_path="./config", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    save_path = get_save_path(cfg)
    setup_seed(cfg.experiment.seed)
    device = get_device(cfg)
    model_class = get_model_class(cfg)
    print(f'The save path is {save_path}')
    logger = setup_logging(save_path)

    # Print current model configuration
    model_name = cfg.models.get("model_name", "unknown_model")
    logger.info(f"Using model configuration: {model_name}")
    logger.info(f"Full config:\n{OmegaConf.to_yaml(cfg)}")
    # Save full config
    config_save_path = save_path / "full_config.yaml"
    OmegaConf.save(config=cfg, f=config_save_path)
    try:
        # Load dataset
        logger.info(f"Loading dataset from {cfg.paths.dataset_path}")
        dataset = CustomDGLDataset(filename=cfg.paths.dataset_path,save_path=cfg.paths.processed_data)
        logger.info(f"Dataset loaded with {len(dataset)} examples")

        hyperparameters = OmegaConf.to_container(cfg, resolve=True)
        if 'models' in hyperparameters and 'model' in hyperparameters['models']:
            hyperparameters.update(hyperparameters['models']['model'])
        print(hyperparameters['batch_size'])
        # Perform K-fold cross-validation
        logger.info("Starting K-fold training...")
        results = train_kfold(
            dataset=dataset,
            model_class=DGNLinearFusion,  
            hyperparameters=hyperparameters,
            device=device,
            n_splits=cfg.experiment.n_splits,
            base_save_dir=save_path
        )

        # Save final results
        with open(save_path / "final_results.json", "w") as f:
            json.dump({
                'average_metrics': results['average_metrics'],
                'std_metrics': results['std_metrics']
            }, f, indent=4)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()