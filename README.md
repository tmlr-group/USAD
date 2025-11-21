# USAD: Unified Statistical Adversarial Detection

Code repository for **CVPR 2026 Submission #16076**

## Repository Structure

```
USAD/
├── README.md                           # This file
├── models/                             # Neural network architectures
│   ├── __init__.py                     # Unified model loader
│   ├── resnet.py                       # ResNet implementation
│   ├── wide_resnet.py                  # Wide-ResNet models
│   ├── swin.py                         # Swin Transformer
│   └── semantic_resnet.py              # Semantic ResNet variant
├── baselines/
│   └── USAD/                           # Core SAD implementation
│       ├── USAD.py                     # Main SAD algorithm
│       └── utils_USAD.py               # Utility functions
├── adv/                                # Adversarial attack generation
│   ├── adv_generator.py                # Attack generation interface
│   ├── attack_generator.py             # Attack implementations
│   ├── checkpoint/                     # Model checkpoints
│   ├── Adv_data/                       # Pre-generated adversarial examples
│   └── scripts/                        # Attack generation scripts
├── exp/                                # Experiments
│   ├── dataloader.py                   # Data loading utilities
│   └── main_exp/                       # Main experiment scripts
│       ├── cifar/res18/                # CIFAR-10 experiments
│       └── imagenet/res50/             # ImageNet experiments
├── data/                               # Dataset storage
│   └── cifar10/                        # CIFAR-10 dataset
└── pretrained/                         # Pretrained model checkpoints
```