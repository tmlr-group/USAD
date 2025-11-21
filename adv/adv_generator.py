import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, vit_b_16
from torch.utils.data import Subset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import *
import numpy as np
import random
import attack_generator_jiacheng as attack
import os

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from resnet18, resnet34, resnet50, vit_b_16")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,imagenet")
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--model_path', default='./checkpoint/CIFAR10/Res18/resnet-18.pth', help='model for white-box attack evaluation')

# Modification: adding configurations of adversarial attacks
parser.add_argument('--epsilon', default=8, type=int, help='perturbation')
parser.add_argument('--num-steps', default=20, type=int, help='perturb number of steps')
parser.add_argument('--num-class', default=10, type=int, help='number of classes')
parser.add_argument('--step-size', default=1, type=int, help='perturb step size')
parser.add_argument('--category', type=str, default='pgd', help='select attack category')
parser.add_argument("--random-start", action="store_true")
parser.add_argument('--norm', type=str, default='linf', help='select attack norm')
parser.add_argument('--stat_names', type=str, default='aggregated', help='select stat names for adpative attack')
args = parser.parse_args()

def main():
    transform_test = transforms.Compose([transforms.ToTensor(),])

    preprocess224 = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()])

    print('==> Load Test Data')
    if args.dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    if args.dataset == "imagenet":
        args.num_class = 1000
        testset = ImageFolder(
            os.path.join("", "val"),
            transform=preprocess224,
        )
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        total_samples = len(testset)
        indices = random.sample(range(total_samples), min(500, total_samples))
        subset = Subset(testset, indices)
        test_loader = torch.utils.data.DataLoader(subset, batch_size=50, shuffle=False, num_workers=0)
        print(f'==> Sampled {len(indices)} samples from ImageNet validation set')

    print('==> Load Model')
    if args.dataset =="cifar10":
        if args.net == "resnet18":
            model =  RN18_10(semantic=False).cuda()
            model = torch.nn.DataParallel(model)
            net = "Res18"
            ckpt = torch.load(args.model_path)
            model.load_state_dict(ckpt)
        if args.net == "wideresnet28":
            args.model_path = "./checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth"
            model = WRN28_10(semantic=False).cuda()
            ckpt = torch.load(args.model_path)
            model.load_state_dict(ckpt['net'])
            model = torch.nn.DataParallel(model)
            net = "WRN28"
        if args.net == "wideresnet70":
            args.model_path = "./checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth"
            model = WRN70_16(semantic=False).cuda()
            ckpt = torch.load(args.model_path)
            model.load_state_dict(ckpt['net'])
            model = torch.nn.DataParallel(model)
            net = "WRN70"
        if args.net == "swin":
            model = swin_t(window_size=4,num_classes=10,downscaling_factors=(2,2,2,1)).cuda()
            model_path = './checkpoint/CIFAR10/SWIN/swin-4.pth'
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['net'])
            model.eval()
            net = "SWIN"
    if args.dataset =="imagenet":
        if args.net == "resnet50":
            model = resnet50(weights="IMAGENET1K_V2").cuda()
            model = torch.nn.DataParallel(model)
            net = "Res50"
        if args.net == "vit_b_16":
            model = vit_b_16(weights="IMAGENET1K_V1").cuda()
            model = torch.nn.DataParallel(model)
            net = "ViT_b_16"

    print(net)

    model.eval()
    print('==> Generate adversarial sample')

    PATH_DATA='./Adv_data/{}/{}/with_labels'.format(args.dataset, net)

    if args.category == 'adaptive':
        ATTACK_FILENAME = 'Adv_{}_{}_{}_eps{}_{}_{}.npz'.format(args.dataset, args.category, args.num_steps, args.epsilon, args.norm, args.stat_names)
    else:
        ATTACK_FILENAME = 'Adv_{}_{}_{}_eps{}_{}.npz'.format(args.dataset, args.category, args.num_steps, args.epsilon, args.norm)

    os.makedirs(PATH_DATA, exist_ok=True)

    args.epsilon = args.epsilon / 255
    args.step_size = args.epsilon / 5

    X_adv, predicted_original_labels, predicted_adv_labels = attack.adv_generate(
        model = model, 
        test_loader = test_loader,
        args = args
    )

    # Save as a tuple: (X_adv, predicted_original_label, predicted_adv_label)
    np.savez(
        os.path.join(PATH_DATA, ATTACK_FILENAME),
        X_adv=X_adv.detach().cpu().numpy(),
        predicted_original_labels=predicted_original_labels.detach().cpu().numpy(),
        predicted_adv_labels=predicted_adv_labels.detach().cpu().numpy(),
    )
    print('Adversarial examples (with labels) saved to: {}'.format(os.path.join(PATH_DATA, ATTACK_FILENAME)))

if __name__ == "__main__":
    main()
