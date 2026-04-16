import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from src.config import FIGURES_DIR
import logging

class GradCAM:
    """
    Grad-CAM for ResNet-50.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()
        
    def generate(self, input_image, class_idx=None):
        self.model.eval()
        
        # Forward
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output).item()
            
        # Backward
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Compute CAM
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        cam = F.relu(cam) # Only positive influence
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

def generate_vit_attention(model, input_image):
    """
    Extracts attention map for ViT-B/16.
    Uses a forward pre-hook to intercept the input to the last MultiheadAttention,
    then manually computes attention weights.
    """
    model.eval()
    attn_inputs = []
    
    def hook(module, args):
        attn_inputs.append(args[0])
        
    # Hook the last attention layer
    target_layer = model.encoder.layers[-1].self_attention
    handle = target_layer.register_forward_pre_hook(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_image)
        
    handle.remove()
    
    # Input to the MHA layer: (Seq_Len, Batch, Embed_Dim) or (Batch, Seq_Len, Embed_Dim)
    # torchvision ViT uses (Batch, Seq_Len, Embed_Dim) but we re-call it to get weights.
    x_ = attn_inputs[0]
    
    # Re-call MHA and request weights for all heads
    with torch.no_grad():
        _, attn_weights = target_layer(x_, x_, x_, need_weights=True, average_attn_weights=False)
    
    # attn_weights shape: (Batch, Num_Heads, Seq_Len, Seq_Len)
    # We want attention from CLS token (index 0) to all other tokens.
    attn_cls = attn_weights[0, :, 0, 1:] # Drop cls token attending to itself
    
    # Average across all heads
    attn_avg = torch.mean(attn_cls, dim=0) # Shape: (Seq_Len - 1)
    
    # Reshape to a 2D grid based on number of patches
    # ViT-B/16 has 14x14 patches for 224x224 image
    grid_size = int(np.sqrt(attn_avg.size(0)))
    attn_grid = attn_avg.reshape(grid_size, grid_size)
    
    # Normalize
    attn_grid = attn_grid - torch.min(attn_grid)
    attn_grid = attn_grid / torch.max(attn_grid)
    
    return attn_grid.cpu().numpy()

def overlay_heatmap(img_tensor, heatmap, save_path=None):
    """
    Upsamples the heatmap and overlays it on the original image.
    img_tensor: (1, 3, H, W) normalized tensor
    heatmap: 2D numpy array
    """
    # Denormalize image for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    
    # Upsample heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    heatmap_colored = heatmap_colored[..., ::-1] # BGR to RGB
    
    # Overlay
    cam_result = heatmap_colored * 0.5 + img * 0.5
    cam_result = np.clip(cam_result, 0, 1)
    
    if save_path:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cam_result)
        plt.title("Attention/Grad-CAM Map")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved visualization to {save_path}")
        
    return cam_result

def generate_robustness_grid(model, model_name, raw_image, corruption_type, device, save_path):
    """
    Applies a sequence of corruptions (Clean -> Severity 5), generates heatmaps, and plots in a 2x3 Grid.
    """
    from src.core.corruptions import apply_corruption
    
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for sev in range(6):
        img_np = np.array(raw_image.resize((224, 224)))
        if sev > 0:
            img_np = apply_corruption(img_np, corruption_type, sev)
            
        img_pil = Image.fromarray(img_np.astype('uint8'))
        img_tensor = post_transform(img_pil).unsqueeze(0).to(device)
        
        if model_name == "resnet50":
            cam = GradCAM(model, target_layer=model.layer4)
            heatmap = cam.generate(img_tensor)
        elif model_name == "vit_b_16":
            heatmap = generate_vit_attention(model, img_tensor)
            
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        base_img = inv_normalize(img_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        base_img = np.clip(base_img, 0, 1)
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255
        heatmap_colored = heatmap_colored[..., ::-1]
        
        cam_result = heatmap_colored * 0.5 + base_img * 0.5
        cam_result = np.clip(cam_result, 0, 1)
        
        title = "Clean" if sev == 0 else f"{corruption_type.capitalize()} (Sev {sev})"
        
        axes[sev].imshow(cam_result)
        axes[sev].set_title(title, fontsize=14)
        axes[sev].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved {corruption_type} grid progression to {save_path}")
