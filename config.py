import os

# Base paths
def _get_base_checkpoint_dir():
    # Detect if running in Colab and if drive is mounted
    if os.path.exists('/content/drive/MyDrive'):
        return '/content/drive/MyDrive/attention-vit/checkpoints'
    return './checkpoints'

# Environment setup
CHECKPOINT_DIR = _get_base_checkpoint_dir()
DATA_DIR = './data'
RESULTS_DIR = './results'
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_CLASSES = 10 # CIFAR-10

# Models list to evaluate
MODELS = ["resnet50", "vit_b_16"]
