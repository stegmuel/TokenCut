import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits
#import moco.vits as vits_moco
from dino import utils
import os


# def get_model(arch, patch_size, device, pretrained_weights, key):
#
#     # Initialize model with pretraining
#     url = None
#     if "moco" in arch:
#         if arch == "moco_vit_small" and patch_size == 16:
#             url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
#         elif arch == "moco_vit_base" and patch_size == 16:
#             url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
#         model = vits.__dict__[arch](num_classes=0)
#     elif "mae" in arch:
#         if arch == "mae_vit_base" and patch_size == 16:
#             url = "mae/visualize/mae_visualize_vit_base.pth"
#         model = vits.__dict__[arch](num_classes=0)
#     elif "vit" in arch:
#         if arch == "vit_small" and patch_size == 16:
#             url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
#         elif arch == "vit_small" and patch_size == 8:
#             url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
#         elif arch == "vit_base" and patch_size == 16:
#             url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
#         elif arch == "vit_base" and patch_size == 8:
#             url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
#         elif arch == "resnet50":
#             url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
#         model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
#
#     if os.path.exists(pretrained_weights):
#         model.cuda()
#         utils.load_pretrained_weights(model, pretrained_weights, key, arch, patch_size)
#
#     for p in model.parameters():
#         p.requires_grad = False
#
#     model.eval()
#     model.to(device)
#     return model


def get_model(arch, patch_size, device, pretrained_weights, key):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False

    # Initialize model with pretraining
    if "imagenet" not in arch:
        model.cuda()
        prefix = None
        key_ = "model"
        if any(s in pretrained_weights for s in ['soco', 'pixpro']):
            prefix = "module.encoder."
        elif "mae" not in pretrained_weights:
            if 'student' in key:
                prefix = "module.backbone."
            else:
                prefix = "backbone."
            key_ = key
        utils.load_pretrained_weights(model, pretrained_weights, key_, prefix)
    model.eval()
    return model
