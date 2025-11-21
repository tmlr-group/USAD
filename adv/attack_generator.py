import numpy as np
from models import *
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchattacks import PGD, PGDL2, AutoAttack, FGSM, BIM
import sys
import os

# Add baselines/SAD directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sad_dir = os.path.join(parent_dir, 'baselines', 'SAD')
sys.path.append(sad_dir)

from SAD_with_gradient import *

# updated version of adv_generate, including an updated version of craft_adv
def adv_generate(
    model, 
    test_loader,
    args
    ):
    print('current attack settings:')
    print('category is: ', args.category)
    print('epsilon is: ', args.epsilon)
    print('step size is: ', args.step_size)
    print('norm is: ', args.norm)
    if args.category != 'fgsm':
        print('num_steps is: ', args.num_steps)
    model.eval()
    bool_i=0
    predicted_original_labels_list = []
    predicted_adv_labels_list = []
    target_list = []
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # collect original target
            target_list.append(target.clone().cpu())
            # replace pgd with craft_adv
            x_adv = craft_adv(
                model = model,
                x_clean = data,
                y = target,
                args = args
            )
            # predict labels on adversarial examples (no grad needed)
            with torch.no_grad():
                logits_adv = model(x_adv)
                logits_ori = model(data)
                preds_adv = torch.argmax(logits_adv, dim=1)
                preds_ori = torch.argmax(logits_ori, dim=1)
            # collect predictions
            predicted_original_labels_list.append(preds_ori.clone().cpu())
            predicted_adv_labels_list.append(preds_adv.clone().cpu())
            if bool_i == 0:
                X_adv = x_adv.clone().cpu()
            else :
                X_adv = torch.cat((X_adv, x_adv.clone().cpu()), 0)
            bool_i +=1
    # concatenate all collected data
    all_targets = torch.cat(target_list, dim=0)
    original_labels = torch.cat(predicted_original_labels_list, dim=0)
    adv_labels = torch.cat(predicted_adv_labels_list, dim=0)
    
    # Filter: only keep samples where original prediction matches ground truth
    correct_mask = (original_labels == all_targets)
    print(f'Total samples: {len(correct_mask)}, Correctly classified: {correct_mask.sum().item()}')
    
    # Apply filter to all data
    X_adv_filtered = X_adv[correct_mask]
    original_labels_filtered = original_labels[correct_mask]
    adv_labels_filtered = adv_labels[correct_mask]
    
    return X_adv_filtered, original_labels_filtered, adv_labels_filtered

# updated version, including more types of adversarial attacks
def craft_adv(
    model,
    x_clean, 
    y, 
    args
    ):
    if args.category == 'pgd':
        attack = PGD(
            model, 
            eps=args.epsilon, 
            alpha=args.step_size, 
            steps=args.num_steps, 
            random_start=args.random_start
            )
    elif args.category == 'pgd-l2':
        attack = PGDL2(
            model, 
            eps=args.epsilon, 
            alpha=args.step_size, 
            steps=args.num_steps, 
            random_start=args.random_start
            )
    elif args.category == 'aa':
        attack = AutoAttack(
            model, 
            norm='Linf', 
            eps=args.epsilon, 
            version='rand',
            n_classes=args.num_class
            )
    elif args.category == 'aa-l2':
        attack = AutoAttack(
            model, 
            norm='L2', 
            eps=args.epsilon, 
            version='rand',
            n_classes=args.num_class
            )
    elif args.category == 'cw-l2':
        adv = CW_l2(
            model, 
            x_clean, 
            y, 
            args.epsilon, 
            args.step_size, 
            args.num_steps,
            args.random_start,
            args.num_class
            )
        return adv
    elif args.category == 'cw':
        adv = CW_linf(
            model, 
            x_clean, 
            y, 
            args.epsilon, 
            args.step_size, 
            args.num_steps, 
            args.random_start,
            args.num_class
            )
        return adv
    elif args.category == 'fgsm':
        attack = FGSM(
            model, 
            eps=args.epsilon,
            )
    elif args.category == 'fgsm-l2':
        adv = FGSM_l2(
            model, 
            x_clean, 
            y, 
            args.epsilon
            )
        return adv
    elif args.category == 'bim':
        attack = BIM(
            model, 
            eps=args.epsilon, 
            alpha=args.step_size, 
            steps=args.num_steps
            )
    elif args.category == 'adaptive':
        x_adv = adaptive_PGD(
            model, 
            x_clean, 
            y, 
            args.epsilon, 
            args.step_size, 
            args.num_steps, 
            args.random_start, 
            args.stat_names
            )
        return x_adv
    x_adv = attack(x_clean, y)
    return x_adv


def cwloss(output, target, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def CW_linf(model, data, target, epsilon, step_size, num_steps, rand_init, num_classes):
    model.eval()
    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            loss_adv = cwloss(output, target, num_classes=num_classes)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def CW_l2(model, data, target, epsilon, step_size, num_steps, rand_init, num_classes):
    model.eval()
    
    # 1. Random initialization within L2 ball
    if rand_init:
        # Sample from uniform distribution in L2 ball
        delta = torch.randn_like(data).cuda()
        # Normalize to unit sphere then scale to epsilon
        delta_flat = delta.view(delta.shape[0], -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        delta_flat = delta_flat / (delta_norm + 1e-8) * epsilon
        # Random scaling: uniformly sample radius
        r = torch.rand(delta.shape[0], 1).cuda() ** (1.0 / delta_flat.shape[1])
        delta = (delta_flat * r).view_as(data)
        x_adv = data.detach() + delta
    else:
        x_adv = data.detach()
    
    # Clamp to valid image range
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    # 2. Iterative attack
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        
        with torch.enable_grad():
            loss_adv = cwloss(output, target, num_classes=num_classes)
        loss_adv.backward()
        
        # 3. L2 gradient update (not sign!)
        grad = x_adv.grad.detach()
        grad_flat = grad.view(grad.shape[0], -1)
        grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
        # Normalize gradient and scale by step_size
        grad_normalized = grad_flat / (grad_norm + 1e-8)
        eta = (grad_normalized * step_size).view_as(grad)
        
        x_adv = x_adv.detach() + eta
        
        # 4. Project back to L2 ball centered at original data
        delta = x_adv - data
        delta_flat = delta.view(delta.shape[0], -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        # If norm > epsilon, project to epsilon-sphere
        delta_flat = delta_flat / torch.clamp(delta_norm / epsilon, min=1.0)
        delta = delta_flat.view_as(data)
        x_adv = data + delta
        
        # 5. Clamp to valid image range [0, 1]
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    return x_adv

def FGSM_l2(model, data, labels, epsilon):
    model.eval()
    
    x_adv = data.detach().clone()
    x_adv.requires_grad = True
    
    output = model(x_adv)
    
    loss = nn.CrossEntropyLoss()
    
    cost = loss(output, labels)
    
    grad = torch.autograd.grad(cost, x_adv, retain_graph=False, create_graph=False)[0]
    
    grad_flat = grad.view(grad.shape[0], -1)
    grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
    grad_normalized = grad_flat / (grad_norm + 1e-8)
    
    perturbation = (grad_normalized * epsilon).view_as(grad)
    x_adv = x_adv.detach() + perturbation
    
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    return x_adv

def adaptive_PGD(model, data, target, epsilon, step_size, num_steps, rand_init, stat_names):
    model.eval()

    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        if stat_names == 'variance':
            USAD_output = STATS(data, x_adv, epsilon, ['variance'])
        elif stat_names == 'uncertainty':
            USAD_output = STATS(data, x_adv, epsilon, ['uncertainty'])
        elif stat_names == 'aggregated':
            USAD_output = STATS(data, x_adv, epsilon, ['variance', 'uncertainty'])
        loss_ce=nn.CrossEntropyLoss()(output, target)
        loss_usad=USAD_output

        model.zero_grad()
        with torch.enable_grad():
            loss_adv = loss_ce - loss_usad

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv