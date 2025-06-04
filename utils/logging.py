import os
import logging
from typing import Optional
from pathlib import Path

def setup_logging(
    log_dir: Optional[str] = None,  # Must be provided from Hydra config (e.g., cfg.paths.log_dir)
    module_name: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    use_timestamp_in_filename: bool = False  # Optional: Whether to add timestamp to log file name
) -> logging.Logger:
    """
    Args:
        log_dir: Must be provided from Hydra config (e.g., cfg.paths.log_dir)
        module_name: Module name (used to distinguish different module logs)
        console_level/file_level: Log level control
        use_timestamp_in_filename: Whether to add timestamp to log file name (default False)
    """
    if log_dir is None:
        raise ValueError("log_dir must be provided from Hydra config!")
    
    # Create log directory (Hydra has already generated parent directory, ensure subdirectory exists)
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Get logger instance
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers repeatedly
    if logger.handlers:
        return logger
        
    # Unified format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (keep unchanged)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # File handler (simplify naming logic)
    log_filename = f"{module_name or 'global'}.log"  # Use module name directly as filename
    log_file = log_dir / log_filename
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger