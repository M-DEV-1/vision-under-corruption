# Interpretability Analysis of CNNs and Vision Transformers under Image Corruptions

Robustness and interpretability analysis of CNNs vs Vision Transformers under image corruptions, with portable training infrastructure.

## Models

| Model     | Architecture | Parameters                |
| --------- | ------------ | ------------------------- |
| ResNet-50 | CNN          | ~25.6M (trainable: ~20K)  |
| ViT-B/16  | Transformer  | ~86.6M (trainable: ~7.7K) |

Both models are pretrained on ImageNet and fine-tuned on CIFAR-100 with frozen backbones.

## Pipeline

```
train → evaluate (clean + corrupted) → interpret (Grad-CAM / Attention)
```

### Training

- Transfer learning with frozen backbone
- Checkpoint saving every epoch

### Robustness Evaluation

- Gaussian Blur (severity 1-5)
- Gaussian Noise (severity 1-5)
- Rotation (severity 1-5)

### Interpretability

- **ResNet-50**: Grad-CAM on final convolutional layer
- **ViT-B/16**: CLS token attention from last encoder block

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run.py --mode all
```

### Individual stages

```bash
python scripts/run.py --mode train --epochs 10
python scripts/run.py --mode eval
python scripts/run.py --mode interpret
```

### Resume training

Training resumes automatically from the latest checkpoint. To start fresh:

```bash
python scripts/run.py --mode train --no-resume
```

## Cross-Platform Usage

| Environment | Command | Checkpoint Storage |
|-------------|---------|-------------------|
| Colab (T4) | `!python scripts/run.py` | Google Drive (auto) |
| Linux / DGX | `python scripts/run.py` | `./checkpoints/` |
| Windows | `python scripts/run.py` | `./checkpoints/` |

## Project Structure

```
attention-vit/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── data/
│   │   └── dataset.py         # CIFAR-100 data pipeline
│   ├── models/
│   │   └── architecture.py    # ResNet-50 and ViT-B/16 loading
│   ├── core/
│   │   ├── train.py           # Training loop + checkpointing
│   │   ├── evaluate.py        # Clean + corrupted evaluation
│   │   └── corruptions.py     # Image corruption functions
│   └── utils/
│       ├── logger.py          # Seed, device, logging
│       └── interpretability.py# Grad-CAM and attention maps
├── scripts/
│   └── run.py                 # Main entry point
├── notebooks/                 
│   └── 01_pipeline.ipynb      # Interactive execution and visualization
├── requirements.txt
└── results/
    ├── tables/                # CSV results (tracked)
    └── figures/               # Generated visualizations (gitignored)
```

## License

MIT
