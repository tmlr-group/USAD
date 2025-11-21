import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision import datasets
import random
import torch
import numpy as np
import builtins
import os

builtins.IS_LOG = True
builtins.DATALOG_COUNT = 0
def log(*args, sep=False, **kwargs):
    if builtins.IS_LOG:
        if sep:
            msg = " ".join(map(str, args))
            total_len = 80
            msg_len = len(msg)
            if msg_len >= total_len:
                print(msg)
            else:
                eq_len = total_len - msg_len
                left = eq_len // 2
                right = eq_len - left
                print(f"{'=' * left}{msg}{'=' * right}")
        else:
            print(*args, **kwargs)

def load_data(path, N, rs, check, model, ref, is_test=True, class_idx=None):
    if 'cifar10' in path:
        samples = sample_CIFAR10(path, N, rs, check, model, ref, is_test, class_idx=class_idx)
    if 'imagenet' in path:
        samples = sample_IMAGENET(path, N, rs, check, model, ref, is_test, class_idx=class_idx)
    return samples

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
import sys
def sample_IMAGENET(path, N, rs, check, model, ref="adv", is_test=True, class_idx=None, batch_size=64):

    device = check_device()
    model = model.to(device) if model is not None else None

    if path.endswith(".npz"):
        ADV, n_total = load_imagenet_adv_success(path)
        if not is_test and builtins.DATALOG_COUNT == 0:
            log(f"Loaded {n_total} total samples, {len(ADV)} successfully attacked samples", sep=True)
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    ADV = torch.from_numpy(ADV).float().to(device)
    if model is not None:
        ADV_rep = rep_by_batch(ADV, model, batch_size)
    else:
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("No model provided, using zero-filled representation", sep=True)
        ADV_rep = torch.zeros_like(ADV)

    # shuffle
    np.random.seed(10086+rs)
    ind = np.random.choice(len(ADV), len(ADV), replace=False)
    ADV, ADV_rep = ADV[ind], ADV_rep[ind]

    if ref == "adv":
        n_samples = len(ADV)
        split_idx = n_samples // 2
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("ADV as ref", sep=True)
            log("ADV.shape: {}".format(ADV.shape))
        if check:
            P = ADV[:split_idx]
            Q = ADV[split_idx:]
            P_rep = ADV_rep[:split_idx]
            Q_rep = ADV_rep[split_idx:]
        else:
            P = ADV #[:split_idx]
            Q = load_imagenet_test(rs).to(device) #[:split_idx].to(device) # Q is shuffled inside
            P_rep = ADV_rep #[:split_idx]
            if model is not None:
                Q_rep = rep_by_batch(Q, model, batch_size)
            else:
                Q_rep = torch.zeros_like(Q)
    
    elif ref == "org":
        CLN = load_imagenet_test(rs).to(device)
        CLN_rep = rep_by_batch(CLN, model, batch_size)
        n_samples = len(CLN)
        split_idx = n_samples // 2
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("ORG as ref", sep=True)
            log("ORG.shape: {}, ADV.shape: {}".format(CLN.shape, ADV.shape))
        if check:
            P = CLN
            Q = ADV
            P_rep = CLN_rep
            Q_rep = ADV_rep
        else:
            P = CLN[:split_idx]
            Q = CLN[split_idx:]
            P_rep = CLN_rep[:split_idx]
            Q_rep = CLN_rep[split_idx:]

    return (flatten_img(P), flatten_img(Q)), (P_rep, Q_rep)

def sample_CIFAR10(path, N, rs, check, model, ref="adv", is_test=True, class_idx=None, batch_size=128):
    """ Default reference set is adv """
    device = check_device()
    model = model.to(device) if model is not None else None
    if path.endswith(".npz"):
        ADV, n_total = load_cifar10_adv_success(path, class_idx=class_idx)
        
        if not is_test and builtins.DATALOG_COUNT == 0:
            if class_idx is not None:
                log(f"Loaded {n_total} total samples, {len(ADV)} successfully attacked samples as class {class_idx}", sep=True)
            else:
                log(f"Loaded {n_total} total samples, {len(ADV)} successfully attacked samples", sep=True)
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    ADV = torch.from_numpy(ADV).float().to(device)
    if model is not None:
        ADV_rep = rep_by_batch(ADV, model, batch_size)
    else:
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("No model provided, using zero-filled representation", sep=True)
        ADV_rep = torch.zeros_like(ADV)

    # shuffle
    np.random.seed(10086+rs)
    ind = np.random.choice(len(ADV), len(ADV), replace=False)
    ADV, ADV_rep = ADV[ind], ADV_rep[ind]
    
    if ref == "adv":
        n_samples = len(ADV)
        split_idx = n_samples // 2
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("ADV as ref", sep=True)
            log("ADV.shape: {}".format(ADV.shape))
        if not check:
            P = ADV[:split_idx]
            Q = ADV[split_idx:]
            P_rep = ADV_rep[:split_idx]
            Q_rep = ADV_rep[split_idx:]
        else:
            P = ADV #[:split_idx]
            Q = load_cifar10_test(class_idx=class_idx).to(device) #[:split_idx].to(device) # Q is shuffled inside
            P_rep = ADV_rep #[:split_idx]
            if model is not None:
                Q_rep = rep_by_batch(Q, model, batch_size)
            else:
                Q_rep = torch.zeros_like(Q)

    elif ref == "org":
        CLN = load_cifar10_test(class_idx=class_idx).to(device)
        CLN_rep = rep_by_batch(CLN, model, batch_size)
        n_samples = len(CLN)
        split_idx = n_samples // 2
        if not is_test and builtins.DATALOG_COUNT == 0:
            log("ORG as ref", sep=True)
            log("ORG.shape: {}, ADV.shape: {}".format(CLN.shape, ADV.shape))
        if check:
            P = CLN
            Q = ADV
            P_rep = CLN_rep #[:split_idx]
            Q_rep = ADV_rep #[:split_idx]
        else:
            P = CLN[:split_idx]
            Q = CLN[split_idx:]
            P_rep = CLN_rep[:split_idx]
            Q_rep = CLN_rep[split_idx:]

    return (flatten_img(P), flatten_img(Q)), (P_rep, Q_rep)

def rep_by_batch(data, model, batch_size):
    n = data.shape[0]
    n_batches = (n - 1) // batch_size + 1
    outputs = []

    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            batch = data[start_idx:end_idx]
            batch_output = model(batch)
            outputs.append(batch_output.cpu())
            torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0).to(data.device)

def flatten_img(img):
    return img.view(img.size(0), -1)

def load_train(path, rs):
    torch.manual_seed(rs)
    if 'cifar10' in path:
        transform = transforms.Compose([transforms.ToTensor(),])
        train = datasets.CIFAR10(root=builtins.CLEAN_PATH, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=True, num_workers=0)
        imgs, labels = next(iter(train_loader))
        # print(labels[:1000])
        return flatten_img(imgs[:5000])
    elif 'imagenet' in path:
        preprocess224 = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()])
        train = ImageFolder(
                os.path.join("/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC/", "train"),
                transform=preprocess224,
            )
        train_loader = torch.utils.data.DataLoader(train, batch_size=5000, shuffle=True, num_workers=0)
        imgs, labels = next(iter(train_loader))
        # print(labels[:1000])
        return flatten_img(imgs[:3000])

def load_cifar10_test(class_idx=None):
    if class_idx is not None:
        transform_test = transforms.Compose([transforms.ToTensor(),])
        testset = datasets.CIFAR10(root=builtins.CLEAN_PATH, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=0)
        imgs, labels = next(iter(test_loader))
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
        imgs = imgs[class_indices]
        return imgs
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = datasets.CIFAR10(root=builtins.CLEAN_PATH, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=0) # shuffle Q
    imgs, _ = next(iter(test_loader))
    return imgs

def load_imagenet_test(rs):
    preprocess224 = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()])
    clean_val_dir = os.path.join(builtins.CLEAN_PATH, "val")
    testset = ImageFolder(
            root=clean_val_dir,
            transform=preprocess224,
        )
    random.seed(rs)
    np.random.seed(rs)
    torch.manual_seed(rs)
    total_samples = len(testset)
    load_n = min(500, total_samples)
    indices = random.sample(range(total_samples), load_n)
    subset = Subset(testset, indices)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=load_n, shuffle=False, num_workers=0)
    imgs, _ = next(iter(test_loader))
    return imgs

def load_cifar10_train(n_c):
    transform_train = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.CIFAR10(root=builtins.CLEAN_PATH, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    imgs, labels = next(iter(train_loader))

    # Select n_c images from each class
    indices = []
    for i in range(10):  # 10 classes in CIFAR-10
        class_indices = (labels == i).nonzero(as_tuple=True)[0][:n_c]
        indices.append(class_indices)
    indices = torch.cat(indices)
    imgs = imgs[indices]
    return imgs

def load_cifar10_train_with_label(model, device, n_c=5000):
    
    class_reps_path = os.path.join(builtins.HELPER_PATH, "class_reps.pt")
    
    # Check if files exist and load them
    if os.path.exists(class_reps_path):
        log(f"Loading class_reps from {class_reps_path}")
        class_reps = torch.load(class_reps_path)
        return class_reps
    
    transform_train = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.CIFAR10(root=builtins.CLEAN_PATH, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    imgs, labels = next(iter(train_loader))
    
    # Initialize dict to store representations for each class
    class_reps = {}
    
    for i in range(10):  # 10 classes in CIFAR-10
        indices = (labels == i).nonzero(as_tuple=True)[0] 
        class_imgs = imgs[indices]
        class_rep = rep_by_batch(class_imgs, model, 128)
        class_reps[i] = [flatten_img(class_imgs).to(device), class_rep]
        log(f"Class {i}: {len(class_imgs)} images processed")
    
    # Save class_reps for future use
    torch.save(class_reps, class_reps_path)
    log(f"Saved class_reps to {class_reps_path}")
            
    return class_reps

def load_imagenet_adv_success(path):
    data = np.load(path)
    X_adv = data['X_adv']
    original_labels = data['predicted_original_labels']
    predicted_labels = data['predicted_adv_labels']

    success_mask = (predicted_labels != original_labels)
    ADV = X_adv[success_mask]
    return ADV, len(X_adv)

def load_cifar10_adv_success(path, class_idx=None):
    data = np.load(path)
    X_adv = data['X_adv']
    original_labels = data['predicted_original_labels']
    predicted_labels = data['predicted_adv_labels']

    success_mask = (predicted_labels != original_labels)
    if class_idx is not None:
        class_mask = (predicted_labels == class_idx)
        mask = success_mask & class_mask
    else:
        mask = success_mask
    ADV = X_adv[mask]
    return ADV, len(X_adv)

def setup_time_log():
    builtins.MMDAGG_TIME_LOG = 0
    builtins.MMDFUSE_TIME_LOG = 0
    builtins.DUAL_TIME_LOG = 0
    builtins.SAD_TIME_LOG = 0
    builtins.EPS_AD_TIME_LOG = 0
    builtins.SAMMD_TIME_LOG = 0
    builtins.FUSE_TIME_LOG = 0
    builtins.AGG_TIME_LOG = 0
    builtins.MMDAgg_TIME_LOG = 0
    builtins.MMD_TIME_LOG = 0
    builtins.MMD_DUAL_TIME_LOG = 0
    builtins.MMD_SAD_TIME_LOG = 0