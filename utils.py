import torch
import numpy as np
import random
import logging
import sys

def set_seed(seed=42):
    """
    Sets seed for reproducibility.
    Handles standard libraries and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """
    Returns the appropriate device (CUDA if available, else CPU).
    Logs the selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA: {torch.version.cuda})")
    else:
        logging.info("Using CPU")
        
    return device

def setup_logging():
    """
    Configures basic logging to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
