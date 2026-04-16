import argparse
import logging
import os
import torch

import sys

# Add project root to path so src modules can be resolved
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.config import *
from src.utils.logger import set_seed, get_device, setup_logging
from src.models.architecture import get_model
from src.data.dataset import get_dataloaders
from src.core.train import train_model
from src.core.evaluate import evaluate_robustness
from src.utils.interpretability import GradCAM, generate_vit_attention, overlay_heatmap, generate_robustness_grid

def main():
    parser = argparse.ArgumentParser(description="Attention-ViT vs CNN Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'interpret', 'all'])
    parser.add_argument('--model', type=str, default='all', choices=['resnet50', 'vit_b_16', 'all'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--no-resume', action='store_true', help="Start training from scratch")
    args = parser.parse_args()

    setup_logging()
    set_seed(42)
    device = get_device()
    
    models_to_run = MODELS if args.model == 'all' else [args.model]
    
    train_loader, val_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    
    # --- TRAINING PHASE ---
    if args.mode in ['train', 'all']:
        logging.info("\n========== PHASE 1: TRAINING ==========")
        for model_name in models_to_run:
            logging.info(f"=== Training Model: {model_name} ===")
            model = get_model(model_name, NUM_CLASSES)
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                device=device,
                checkpoint_dir=CHECKPOINT_DIR,
                model_name=model_name,
                learning_rate=LEARNING_RATE,
                no_resume=args.no_resume
            )
            
    # --- EVALUATION PHASE ---
    if args.mode in ['eval', 'all']:
        logging.info("\n========== PHASE 2: ROBUSTNESS EVALUATION ==========")
        for model_name in models_to_run:
            logging.info(f"=== Evaluating Model: {model_name} ===")
            model = get_model(model_name, NUM_CLASSES)
            # We must load latest checkpoint before evaluation
            from src.core.train import load_checkpoint
            _ = load_checkpoint(model, None, CHECKPOINT_DIR, model_name, device)
            
            model = model.to(device)
            evaluate_robustness(model, DATA_DIR, model_name, device, BATCH_SIZE)
            
    # --- INTERPRETABILITY PHASE ---
    if args.mode in ['interpret', 'all']:
        logging.info("\n========== PHASE 3: INTERPRETABILITY GENERATION ==========")
        for model_name in models_to_run:
            logging.info(f"=== Generating Grids for Model: {model_name} ===")
            model = get_model(model_name, NUM_CLASSES)
            from src.core.train import load_checkpoint
            _ = load_checkpoint(model, None, CHECKPOINT_DIR, model_name, device)
            
            model = model.to(device)
            model.eval()
            
            # Fetch a raw un-transformed validation sample
            from src.data.dataset import get_caltech101_splits
            _, raw_val_dataset, _ = get_caltech101_splits(DATA_DIR, transform=None)
            raw_image, label_idx = raw_val_dataset[0]
            label_name = classes[label_idx]
            
            # We will generate a progression grid for 'blur' as the showcase
            save_path = os.path.join(FIGURES_DIR, f"{model_name}_{label_name}_blur_grid.png")
            generate_robustness_grid(model, model_name, raw_image, 'blur', device, save_path)
            
            # Generate a progression grid for 'noise'
            save_path_noise = os.path.join(FIGURES_DIR, f"{model_name}_{label_name}_noise_grid.png")
            generate_robustness_grid(model, model_name, raw_image, 'noise', device, save_path_noise)

if __name__ == "__main__":
    main()
