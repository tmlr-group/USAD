from .resnet import *
from .wide_resnet import *
from .swin import *
import torch
from torchvision.models import resnet50, vit_b_16

def load_model(model_arch, model_path, semantic=True):
    if "Res18" in model_arch:
        model = RN18_10(semantic=semantic)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    
    if "WRN28" in model_arch:
        # model_path = "./checkpoint/CIFAR10/WRN28/wide-resnet-28x10.pth"
        model = WRN28_10(semantic=semantic)
        model.load_state_dict(torch.load(model_path)['net'])
        model = torch.nn.DataParallel(model)
        model.eval()

    if "WRN70" in model_arch:
        # model_path = "./checkpoint/CIFAR10/WRN70/wide-resnet-70x16.pth"
        model = WRN70_16(semantic=semantic)
        model.load_state_dict(torch.load(model_path)['net'])
        model = torch.nn.DataParallel(model)
        model.eval()
    
    if "SWIN" in model_arch:
        model = swin_t(window_size=4,num_classes=10,downscaling_factors=(2,2,2,1))
        # model_path = './checkpoint/CIFAR10/SWIN/swin-4.pth'
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['net'])
        model.eval()

    if "Res50" in model_arch:
        model = resnet50(weights="IMAGENET1K_V2").cuda()
        # if semantic:
        #     model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        model.eval()
    
    if "ViT_b_16" in model_arch:
        model = vit_b_16(weights="IMAGENET1K_V1").cuda()
        model = torch.nn.DataParallel(model)
        model.eval()

    return model

    