#!/bin/bash

# Script to run adversarial example generator
# Usage: bash run_adv_generator.sh

# Navigate to the adv directory
cd "$(dirname "$0")/.." || exit 1

# Default parameters
dataset='cifar10'
net='resnet18'
steps=5
eps=8

python adv_generator.py --category 'pgd' --dataset $dataset --net $net --epsilon $eps --num-steps $steps --random-start --norm 'linf'
python adv_generator.py --category 'pgd-l2' --dataset $dataset --net $net --epsilon $eps --num-steps $steps --random-start --norm 'l2'
python adv_generator.py --category 'aa' --dataset $dataset --net $net --epsilon $eps --num-steps $steps --norm 'linf'
python adv_generator.py --category 'aa-l2' --dataset $dataset --net $net --epsilon $eps --num-steps $steps --norm 'l2'
python adv_generator.py --category 'cw' --dataset $dataset --net $net --epsilon $eps --num-steps $steps --norm 'linf'
python adv_generator.py --category 'fgsm' --dataset $dataset --net $net --epsilon $eps --norm 'linf'
python adv_generator.py --category 'bim' --dataset $dataset --net $net --epsilon $eps --num-steps $steps --norm 'linf'