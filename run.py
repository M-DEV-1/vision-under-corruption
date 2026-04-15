import argparse
import logging
import os
import torch

from config import *
from utils import set_seed, get_device, setup_logging
from models import get_model
from dataset import get_dataloaders
from train import train_model
from evaluate import evaluate_robustness
from interpretability import GradCAM, generate_vit_attention, overlay_heatmap

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
    
    for model_name in models_to_run:
        logging.info(f"=== Processing Model: {model_name} ===")
        
        # Load Architecture
        model = get_model(model_name, NUM_CLASSES)
        
        # --- TRAINING ---
        if args.mode in ['train', 'all']:
            logging.info(f"Starting Training for {args.epochs} epochs...")
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
            
        # --- EVALUATION ---
        if args.mode in ['eval', 'all']:
            logging.info("Starting Robustness Evaluation...")
            # We must load latest checkpoint before evaluation
            from train import load_checkpoint
            # Dummy optimizer since we only need the model weights for eval
            import torch.optim as optim
            dummy_opt = optim.Adam(model.parameters())
            _ = load_checkpoint(model, dummy_opt, CHECKPOINT_DIR, model_name, device)
            
            model = model.to(device)
            evaluate_robustness(model, DATA_DIR, model_name, device, BATCH_SIZE)
            
        # --- INTERPRETABILITY ---
        if args.mode in ['interpret', 'all']:
            logging.info("Starting Interpretability Generation...")
            from train import load_checkpoint
            import torch.optim as optim
            dummy_opt = optim.Adam(model.parameters())
            _ = load_checkpoint(model, dummy_opt, CHECKPOINT_DIR, model_name, device)
            
            model = model.to(device)
            model.eval()
            
            # Fetch a single batch from validations
            inputs, labels = next(iter(val_loader))
            inputs = inputs.to(device)
            
            # Use the first image in the batch
            single_img = inputs[0:1]
            label_name = classes[labels[0].item()]
            
            save_path = os.path.join(FIGURES_DIR, f"{model_name}_{label_name}_cam.png")
            
            if model_name == "resnet50":
                # ResNet uses Grad-CAM on layer4
                cam = GradCAM(model, target_layer=model.layer4)
                heatmap = cam.generate(single_img)
                overlay_heatmap(single_img, heatmap, save_path)
                
            elif model_name == "vit_b_16":
                # ViT uses Attention extraction
                heatmap = generate_vit_attention(model, single_img)
                overlay_heatmap(single_img, heatmap, save_path)

if __name__ == "__main__":
    main()
