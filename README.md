# USAD: Unified Statistical Adversarial Detection

Official implementation for the ICML 2026 Workshop on Hypothesis Testing paper:

**Unified Statistical Adversarial Detection via Multi-Statistic Two-Sample Testing**

[[OpenReview]](https://openreview.net/forum?id=LoG8Wz4Bga)

## Overview

USAD is a training-free adversarial example detection framework based on kernel two-sample testing. Given a batch of inputs, USAD tests whether the batch has been adversarially perturbed by comparing it against clean reference data in the neural network's representation space. The method aggregates multiple complementary test statistics -- mean divergence (MMD), variance divergence, and uncertainty divergence (covariance-based) -- via permutation testing with controlled Type-I error.

## Repository Structure

```
USAD/
├── baselines/USAD/
│   ├── USAD.py              # Core SAD algorithm (permutation test loop)
│   └── utils_USAD.py        # Kernel functions, MMD, covariance estimation, projection
├── models/
│   ├── __init__.py           # Unified model loader
│   ├── resnet.py             # ResNet-18
│   ├── wide_resnet.py        # WideResNet-28-10, WideResNet-70-16
│   ├── swin.py               # Swin Transformer
│   └── semantic_resnet.py    # Semantic ResNet variant
├── adv/
│   ├── adv_generator.py      # Adversarial example generation pipeline
│   ├── attack_generator.py   # Attack implementations (PGD, FGSM, BIM, CW, AutoAttack, adaptive)
│   └── scripts/              # Shell scripts for batch generation
├── exp/
│   ├── dataloader.py         # Data loading for CIFAR-10 and ImageNet
│   └── main_exp/
│       ├── cifar/res18/      # CIFAR-10 experiments (PGD, FGSM, BIM, CW, AA)
│       └── imagenet/res50/   # ImageNet experiments (PGD, FGSM, BIM, CW, AA)
├── data/                     # Dataset storage
└── pretrained/               # Pretrained model checkpoints
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- torchattacks
- NumPy
- SciPy

## Usage

### 1. Generate Adversarial Examples

```bash
cd adv
python adv_generator.py \
    --net resnet18 \
    --dataset cifar10 \
    --category pgd \
    --epsilon 8 \
    --num-steps 20
```

Supported attacks: `pgd`, `fgsm`, `bim`, `cw`, `aa` (AutoAttack), `adaptive`, and their L2 variants.

### 2. Run Detection Experiments

```bash
cd exp/main_exp/cifar/res18
python cifar_pgd.py \
    --check 1 \
    --N_rf 50 \
    --N_ip 50 \
    --n_test 100 \
    --n_per 100 \
    --alpha 0.05
```

Key arguments:
- `--check 1`: evaluate test power (detection rate); `--check 0`: evaluate Type-I error
- `--N_rf` / `--N_ip`: number of reference / input samples
- `--n_per`: number of permutations for the permutation test
- `--alpha`: significance level
- `--reduction_method`: dimensionality reduction for covariance features (`pca`, `svd`, `random`)

Results (mean and standard error over `--n_exp` repetitions) are saved to `exp/main_exp/.../Results/`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{usad2026,
  title={Unified Statistical Adversarial Detection via Multi-Statistic Two-Sample Testing},
  booktitle={ICML 2026 Workshop on Hypothesis Testing},
  year={2026}
}
```

## License

This project is for research purposes.
