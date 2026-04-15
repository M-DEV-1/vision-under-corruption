import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_resnet50(num_classes):
    """
    Loads pretrained ResNet-50, freezes the backbone, and replaces the final CLS layer.
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze backbone parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final classification layer (unfrozen by default)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_vit_b_16(num_classes):
    """
    Loads pretrained ViT-B/16, freezes the backbone, and replaces the final CLS head.
    """
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    
    # Freeze backbone parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final classification head (unfrozen by default)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_model(model_name, num_classes):
    """
    Returns the requested model architecture.
    """
    if model_name.lower() == "resnet50":
        return get_resnet50(num_classes)
    elif model_name.lower() in ["vit_b_16", "vit-b-16", "vit_b_16"]:
        return get_vit_b_16(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
