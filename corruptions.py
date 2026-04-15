import numpy as np
import cv2
import random

def apply_gaussian_blur(image, severity=1):
    """
    Applies Gaussian Blur corruption.
    Severity [1-5] maps to kernel sizes.
    image: numpy array (H, W, C)
    """
    # Severity to kernel size mapping
    kernels = {1: (3, 3), 2: (5, 5), 3: (7, 7), 4: (9, 9), 5: (11, 11)}
    kernel_size = kernels.get(severity, (3, 3))
    
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred

def apply_gaussian_noise(image, severity=1):
    """
    Applies Additive Gaussian Noise.
    Severity [1-5] maps to noise variance.
    image: numpy array (H, W, C), assumes values in [0, 255] or [0, 1]
    """
    # Std deviations mapping
    std_devs = [0.04, 0.08, 0.12, 0.18, 0.26]
    std = std_devs[severity - 1]
    
    # Check if image is normalized
    max_val = 255.0 if image.dtype == np.uint8 or np.max(image) > 1.5 else 1.0
    
    noise = np.random.normal(0, std * max_val, image.shape)
    noisy_image = np.clip(image + noise, 0, max_val)
    
    return noisy_image.astype(image.dtype)

def apply_rotation(image, severity=1):
    """
    Applies Rotation corruption.
    Severity [1-5] maps to rotation angles.
    image: numpy array (H, W, C)
    """
    angles = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    angle = angles.get(severity, 10)
    
    # Randomly rotate left or right
    angle = angle * random.choice([-1, 1])
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    return rotated

def apply_corruption(image, corruption_type, severity=1):
    """
    Router for corruptions.
    """
    if corruption_type == 'blur':
        return apply_gaussian_blur(image, severity)
    elif corruption_type == 'noise':
        return apply_gaussian_noise(image, severity)
    elif corruption_type == 'rotation':
        return apply_rotation(image, severity)
    else:
        return image
