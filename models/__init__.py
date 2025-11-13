from .resnet import *
import torch
from torchvision.models import resnet50, vit_b_16

def load_model(model_arch, model_path, semantic=True):
    if "Res18" in model_arch:
        model = RN18_10(semantic=semantic)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    if "Res50" in model_arch:
        model = resnet50(weights="IMAGENET1K_V2").cuda()
        model = torch.nn.DataParallel(model)
        model.eval()
    
    if "ViT_b_16" in model_arch:
        model = vit_b_16(weights="IMAGENET1K_V1").cuda()
        model = torch.nn.DataParallel(model)
        model.eval()

    return model

    