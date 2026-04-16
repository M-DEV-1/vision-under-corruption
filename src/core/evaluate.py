import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from src.config import TABLES_DIR, FIGURES_DIR
from src.core.corruptions import apply_corruption
from PIL import Image

class CorruptedDataset(torch.utils.data.Dataset):
    """
    Wrapper for dataset that applies corruptions dynamically.
    """
    def __init__(self, dataset, corruption_type=None, severity=1):
        self.dataset = dataset
        self.corruption_type = corruption_type
        self.severity = severity
        
        # Post-corruption transforms
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We assume the underlying dataset returns (PIL Image, label) before any transform
        # For CIFAR-10, we'll bypass its default transform
        img, label = self.dataset[idx]
        
        # Convert to RGB to ensure 3 channels for ImageNet models
        img = img.convert('RGB')
        
        # Resize to 224x224 (required for models)
        img = img.resize((224, 224))
        
        # Convert to numpy for corruption
        img_np = np.array(img)
        
        if self.corruption_type:
            img_np = apply_corruption(img_np, self.corruption_type, self.severity)
            
        # Back to PIL for ToTensor and Normalize
        img_pil = Image.fromarray(img_np.astype('uint8'))
        
        # Final tensor
        img_tensor = self.post_transform(img_pil)
        
        return img_tensor, label

def evaluate_robustness(model, data_dir, model_name, device, batch_size=128):
    """
    Evaluates a model across all corruptions and severities.
    Outputs results to CSV.
    """
    model.eval()
    
    # Load raw validation dataset (no transforms initially)
    from src.data.dataset import get_caltech101_splits
    _, raw_val_dataset, _ = get_caltech101_splits(data_dir, transform=None)
    
    corruptions = ['clean', 'blur', 'noise', 'rotation']
    severities = [1, 2, 3, 4, 5]
    
    results = []
    
    for corp in corruptions:
        sevs = [1] if corp == 'clean' else severities
        
        for sev in sevs:
            logging.info(f"Evaluating {model_name} on {corp} (Severity: {sev})")
            
            c_type = None if corp == 'clean' else corp
            dataset = CorruptedDataset(raw_val_dataset, corruption_type=c_type, severity=sev)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            correct = 0
            total = 0
            confidences = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"{corp} s{sev}", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    # Store probabilities
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confs, preds = torch.max(probs, 1)
                    
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    confidences.extend(confs.cpu().numpy().tolist())
                    
            accuracy = 100 * correct / total
            avg_confidence = np.mean(confidences)
            
            results.append({
                'Model': model_name,
                'Corruption': corp,
                'Severity': sev,
                'Accuracy': accuracy,
                'Avg_Confidence': avg_confidence
            })
            
            logging.info(f"Accuracy: {accuracy:.2f}%, Avg Conf: {avg_confidence:.4f}")
            
            # Incremental save to prevent data loss on crashes
            df = pd.DataFrame(results)
            os.makedirs(TABLES_DIR, exist_ok=True)
            csv_path = os.path.join(TABLES_DIR, f"{model_name}_robustness_results.csv")
            df.to_csv(csv_path, index=False)

    logging.info(f"Final robustness evaluation saved to {csv_path}")
    
    # Generate Accuracy Dropoff Plot (Instant inference plotting, no training)
    _plot_robustness_dropoff(df, model_name)
    
    return df

def _plot_robustness_dropoff(df, model_name):
    """Generates a line plot showing accuracy dropoff per corruption type."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    clean_acc = df[df['Corruption'] == 'clean']['Accuracy'].values[0]
    plt.axhline(y=clean_acc, color='black', linestyle='--', label=f'Clean Baseline ({clean_acc:.2f}%)')
    
    corruptions = [c for c in df['Corruption'].unique() if c != 'clean']
    for corruption in corruptions:
        subset = df[df['Corruption'] == corruption]
        plt.plot(subset['Severity'], subset['Accuracy'], marker='o', linewidth=2, label=corruption.capitalize())
        
    plt.title(f"{model_name.upper()} - Robustness Accuracy Dropoff", fontsize=16, fontweight='bold')
    plt.xlabel("Corruption Severity", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend(title="Corruption Type")
    plt.ylim(0, 100)
    plt.tight_layout()
    
    plot_path = os.path.join(FIGURES_DIR, f"{model_name}_robustness_dropoff.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved dropoff plot to {plot_path}")
