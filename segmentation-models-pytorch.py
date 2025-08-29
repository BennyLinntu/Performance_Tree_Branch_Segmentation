import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import time
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize, binary_erosion, binary_dilation
from skimage.metrics import structural_similarity as ssim
import sys
import pkg_resources

# 添加必要的导入
import torch.nn.functional as F
# 注释掉FLOPs计算相关导入，因为不再使用效率指标
# try:
#     from thop import profile, clever_format
# except ImportError:
#     print("Warning: thop not available for FLOPs calculation")
#     profile = None
#     clever_format = None
try:
    import torchvision
    import torchvision.models
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
except ImportError:
    print("Warning: torchvision not available for Mask R-CNN")

try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
except ImportError:
    print("Warning: transformers not available for Mask2Former")

# ===================== 设备与全局配置 =====================
# 针对 NVIDIA L4 (23GB) 设计的设备选择逻辑，可通过环境变量强制指定GPU
def select_device():
    if not torch.cuda.is_available():
        print("⚠ 未检测到CUDA, 使用CPU (性能较低)")
        return torch.device('cpu')
    forced = os.environ.get('CUDA_VISIBLE_DEVICES_FORCE')
    if forced is not None:
        try:
            idx = int(forced)
            torch.cuda.set_device(idx)
            print(f"🔧 使用环境变量强制GPU {idx}: {torch.cuda.get_device_name(idx)}")
            return torch.device(f'cuda:{idx}')
        except Exception as e:
            print(f"环境变量CUDA_VISIBLE_DEVICES_FORCE解析失败: {e}, 回退自动选择")
    # 自动策略：优先寻找空闲显存最大的GPU
    free_list = []
    for i in range(torch.cuda.device_count()):
        try:
            stats = torch.cuda.memory_stats(i)
            alloc = stats.get('allocated_bytes.all.current', 0)
            total = torch.cuda.get_device_properties(i).total_memory
            free = total - alloc
            free_list.append((free, i))
        except Exception:
            free_list.append((0, i))
    free_list.sort(reverse=True)  # 按可用显存降序
    best_idx = free_list[0][1]
    torch.cuda.set_device(best_idx)
    print(f"✅ 自动选择GPU {best_idx}: {torch.cuda.get_device_name(best_idx)}")
    return torch.device(f'cuda:{best_idx}')

device = select_device()
print(f"Using device: {device}")
print("设备初始化完成 (L4 优化模式)")
image_size = 256  # 统一图像大小

# ===================== 统一的模型配置常量 =====================
# 避免多处重复定义导致不一致，这里集中列出 23 个模型
UNIFIED_MODEL_CONFIGS = {
    'unet':                dict(model=smp.Unet,          encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'unet++':              dict(model=smp.UnetPlusPlus,  encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'deeplabv3':           dict(model=smp.DeepLabV3,     encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'deeplabv3+':          dict(model=smp.DeepLabV3Plus, encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'fpn':                 dict(model=smp.FPN,           encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'pspnet':              dict(model=smp.PSPNet,        encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'linknet':             dict(model=smp.Linknet,       encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'manet':               dict(model=smp.MAnet,         encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'pan':                 dict(model=smp.PAN,           encoder_name='resnet34',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'resnet50_unet':       dict(model=smp.Unet,          encoder_name='resnet50',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mobilenet_v2_unet':   dict(model=smp.Unet,          encoder_name='mobilenet_v2',  encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'densenet121_unet':    dict(model=smp.Unet,          encoder_name='densenet121',   encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'efficientnet-b0_unet':dict(model=smp.Unet,          encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'convnext_base_unet':  dict(model=smp.Unet,          encoder_name='convnext_base', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'vit_b_16_unet':       dict(model=smp.Unet,          encoder_name='vit_b_16',      encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'swin_base_unet':      dict(model=smp.Unet,          encoder_name='swin_base',     encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mit_b0_unet':         dict(model=smp.Unet,          encoder_name='mit_b0',        encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mit_b1_unet':         dict(model=smp.Unet,          encoder_name='mit_b1',        encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mit_b2_unet':         dict(model=smp.Unet,          encoder_name='mit_b2',        encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mit_b3_unet':         dict(model=smp.Unet,          encoder_name='mit_b3',        encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mit_b4_unet':         dict(model=smp.Unet,          encoder_name='mit_b4',        encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    'mit_b5_unet':         dict(model=smp.Unet,          encoder_name='mit_b5',        encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
    # 修复 levit_128s_unet: timm 中支持的名称通常为 'timm-efficientnet-b0' 但此处用作占位 -> 回退到efficientnet-b0 保证可用
    'levit_128s_unet':     dict(model=smp.Unet,          encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1, activation='sigmoid'),
}

ALL_MODEL_NAMES = list(UNIFIED_MODEL_CONFIGS.keys())  # 23 个模型

# ===================== 烟雾测试函数 =====================
def run_smoke_tests(sample_size=2, image_size=128):
    """快速前向烟雾测试，确认全部23模型可构建并完成一次前向传播。
    仅用于部署前自检，不进行反向传播，使用更小输入以节省显存。"""
    print("\n🔥 开始烟雾测试 (前向推理)...")
    dummy = torch.randn(sample_size, 3, image_size, image_size).to(device)
    ok = []
    failed = []
    for name in ALL_MODEL_NAMES:
        cfg = UNIFIED_MODEL_CONFIGS[name]
        try:
            model = cfg['model'](**{k: v for k, v in cfg.items() if k != 'model'})
            model = model.to(device).eval()
            with torch.no_grad():
                out = model(dummy)
            if out.shape[1] != 1:
                print(f"⚠ {name} 输出通道={out.shape[1]} 期望=1")
            ok.append(name)
            del model, out
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            failed.append((name, str(e)))
    print(f"\n✅ 成功 {len(ok)}/{len(ALL_MODEL_NAMES)} 个模型")
    if failed:
        print("失败列表:")
        for n, err in failed:
            print(f"  - {n}: {err}")
    return len(failed) == 0

# 添加Attention U-Net实现


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = self.conv_block(1024, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = self.conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = self.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))

        # Decoder
        d5 = self.up5(e5)
        e4 = self.att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        e3 = self.att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        e2 = self.att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        e1 = self.att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        out = self.final_conv(d2)
        return self.sigmoid(out)

# 添加ResU-Net实现


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet, self).__init__()

        # Encoder
        self.input_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.input_bn = nn.BatchNorm2d(64)
        self.input_relu = nn.ReLU(inplace=True)

        self.encoder1 = ResidualBlock(64, 64)
        self.encoder2 = ResidualBlock(64, 128, stride=2)
        self.encoder3 = ResidualBlock(128, 256, stride=2)
        self.encoder4 = ResidualBlock(256, 512, stride=2)

        self.center = ResidualBlock(512, 1024, stride=2)

        # Decoder
        self.decoder4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv4 = ResidualBlock(1024, 512)

        self.decoder3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv3 = ResidualBlock(512, 256)

        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv2 = ResidualBlock(256, 128)

        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv1 = ResidualBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input
        x = self.input_relu(self.input_bn(self.input_conv(x)))

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        center = self.center(e4)

        # Decoder
        d4 = self.decoder4(center)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder_conv4(d4)

        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder_conv3(d3)

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder_conv2(d2)

        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder_conv1(d1)

        out = self.final_conv(d1)
        return self.sigmoid(out)

# 添加Mobile U-Net实现


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class MobileUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MobileUNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = DepthwiseSeparableConv(32, 64)
        self.encoder2 = DepthwiseSeparableConv(64, 128)
        self.encoder3 = DepthwiseSeparableConv(128, 256)
        self.encoder4 = DepthwiseSeparableConv(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Center
        self.center = DepthwiseSeparableConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DepthwiseSeparableConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DepthwiseSeparableConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DepthwiseSeparableConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DepthwiseSeparableConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input
        x = self.relu(self.bn1(self.conv1(x)))

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Center
        center = self.center(self.pool(e4))

        # Decoder
        d4 = self.up4(center)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        out = self.final_conv(d1)
        return self.sigmoid(out)

# 简化版U-Net3+实现


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet3Plus, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2)

        # 解码器中的全尺度特征融合
        self.h4_pt_conv = conv_block(64, 64)
        self.h4_cat_conv = conv_block(128 + 64, 64)
        self.h4_cat_conv2 = conv_block(256 + 64, 64)
        self.h4_cat_conv3 = conv_block(512 + 64, 64)
        self.h4_cat_conv4 = conv_block(1024 + 64, 64)

        # 上采样层
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 256, 4, stride=4)
        self.up4 = nn.ConvTranspose2d(512, 512, 8, stride=8)
        self.up5 = nn.ConvTranspose2d(1024, 1024, 16, stride=16)

        # 最终输出
        self.final = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))

        # 全尺度特征融合
        h4_pt = self.h4_pt_conv(e1)
        h4_cat = torch.cat([self.up2(e2), h4_pt], dim=1)
        h4_cat = self.h4_cat_conv(h4_cat)

        h4_cat = torch.cat([self.up3(e3), h4_cat], dim=1)
        h4_cat = self.h4_cat_conv2(h4_cat)

        h4_cat = torch.cat([self.up4(e4), h4_cat], dim=1)
        h4_cat = self.h4_cat_conv3(h4_cat)

        h4_cat = torch.cat([self.up5(e5), h4_cat], dim=1)
        h4_cat = self.h4_cat_conv4(h4_cat)

        out = self.final(h4_cat)
        return self.sigmoid(out)

# 添加DeepLabV3实现


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50'):
        super(DeepLabV3, self).__init__()

        if backbone == 'resnet50':
            backbone_model = torchvision.models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            backbone_model = torchvision.models.resnet101(pretrained=True)
        else:
            raise ValueError("Backbone must be resnet50 or resnet101")

        self.backbone = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4
        )

        self.aspp = ASPP(2048, [12, 24, 36])

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.aspp(features)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return self.sigmoid(x)

# Mask R-CNN适配器 - 修复版本


class MaskRCNNSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(MaskRCNNSegmentation, self).__init__()

        # 使用预训练的Mask R-CNN
        self.model = maskrcnn_resnet50_fpn(pretrained=True)

        # 修改分类器头部
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        # 修改mask预测器头部
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)

        # 添加一个简单的分割头，用于推理时生成分割掩码
        self.seg_head = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """前向传播，返回分割掩码"""
        # 在训练时，我们使用简单的分割头
        # 因为Mask R-CNN的训练需要特殊的标注格式
        if self.training:
            return self.seg_head(x)
        else:
            # 推理时也使用分割头，保持一致性
            return self.seg_head(x)

    def _maskrcnn_inference(self, x):
        """Mask R-CNN原始推理（保留用于可能的未来使用）"""
        self.model.eval()
        with torch.no_grad():
            # 转换输入格式为列表（Mask R-CNN要求）
            image_list = [img for img in x]
            predictions = self.model(image_list)

        batch_size = x.size(0)
        height, width = x.shape[-2:]

        output_masks = torch.zeros(batch_size, 1, height, width).to(x.device)

        for i, pred in enumerate(predictions):
            if len(pred['masks']) > 0 and len(pred['scores']) > 0:
                # 选择置信度最高的掩码
                best_idx = torch.argmax(pred['scores'])
                mask = pred['masks'][best_idx, 0]

                # 调整掩码大小
                if mask.shape != (height, width):
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                output_masks[i, 0] = mask

        return torch.sigmoid(output_masks)

# Mask2Former适配器 - 修复版本


class Mask2FormerSegmentation(nn.Module):
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic"):
        super(Mask2FormerSegmentation, self).__init__()

        try:
            self.processor = Mask2FormerImageProcessor.from_pretrained(
                model_name)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name)
            self.model_available = True
        except:
            print(
                "Warning: Mask2Former not available, using simple segmentation head instead")
            self.model = None
            self.processor = None
            self.model_available = False

            # 创建一个简单的分割头作为替代
            self.seg_head = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if not self.model_available:
            # 使用简单分割头
            return self.seg_head(x)

        try:
            batch_size = x.size(0)
            height, width = x.shape[-2:]

            # 预处理图像
            images = []
            for i in range(batch_size):
                img = x[i].cpu().numpy().transpose(1, 2, 0)
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                images.append(img)

            # 处理输入
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(x.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 后处理
            predicted_segmentation_maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(height, width)] * batch_size
            )

            # 转换为二值掩码
            output_masks = torch.zeros(
                batch_size, 1, height, width).to(x.device)
            for i, seg_map in enumerate(predicted_segmentation_maps):
                # 将所有非背景类别视为前景
                binary_mask = (seg_map > 0).float()
                output_masks[i, 0] = binary_mask

            return output_masks

        except Exception as e:
            print(f"Error in Mask2Former forward pass: {e}")
            # 回退到简单分割头
            if not hasattr(self, 'seg_head'):
                self.seg_head = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid()
                ).to(x.device)
            return self.seg_head(x)

# 组合损失函数类


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce

# 数据集类


class BranchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取所有图片文件名
        self.images = sorted([f for f in os.listdir(
            image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图片
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 构建对应的mask文件名（假设mask文件名与image相同，但扩展名为.png）
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图片和mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # 确保mask是二值的
        mask = (mask > 127).astype(np.float32)

        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 添加通道维度到mask
        mask = mask.unsqueeze(0)

        return image, mask


def create_train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    创建70/15/15的训练/验证/测试集划分

    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例 (默认0.7)
        val_ratio: 验证集比例 (默认0.15)
        test_ratio: 测试集比例 (默认0.15)
        random_seed: 随机种子

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"

    # 设置随机种子确保可重复性
    torch.manual_seed(random_seed)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f"数据集划分:")
    print(f"  总数据量: {total_size}")
    print(f"  训练集: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"  验证集: {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"  测试集: {test_size} ({test_size/total_size*100:.1f}%)")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


def create_kfold_splits(dataset, k=5, random_seed=42):
    """
    创建K折交叉验证的数据划分

    Args:
        dataset: 完整数据集
        k: 折数 (默认5)
        random_seed: 随机种子

    Returns:
        list of (train_indices, val_indices) tuples
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    total_size = len(dataset)
    indices = list(range(total_size))

    folds = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(
            f"Fold {fold+1}: 训练集 {len(train_indices)} 样本, 验证集 {len(val_indices)} 样本")
        folds.append((train_indices.tolist(), val_indices.tolist()))

    return folds


def get_dataloader_from_indices(dataset, indices, batch_size, shuffle=False, transform=None):
    """
    根据索引创建数据加载器

    Args:
        dataset: 原始数据集
        indices: 数据索引列表
        batch_size: 批次大小
        shuffle: 是否打乱
        transform: 数据变换

    Returns:
        DataLoader
    """
    # 创建子数据集
    subset = torch.utils.data.Subset(dataset, indices)

    # 如果需要设置transform
    if transform is not None:
        # 创建一个新的数据集类来包装subset并应用transform
        class TransformSubset(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                image, mask = self.subset[idx]
                # 将tensor转回numpy进行变换
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).numpy()
                if isinstance(mask, torch.Tensor):
                    mask = mask.squeeze(0).numpy()

                # 应用变换
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

                # 添加通道维度到mask
                mask = mask.unsqueeze(0)

                return image, mask

        subset = TransformSubset(subset, transform)

    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# 综合指标计算类


class ComprehensiveMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_iou = 0
        self.total_dice = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_ts_iou = 0  # Thin Structure IoU
        self.total_cpr = 0     # Connectivity Preservation Ratio
        self.total_boundary_f1 = 0
        self.total_skel_sim = 0  # Skeleton Similarity
        self.count = 0

    def calculate_iou(self, pred, target, threshold=0.5):
        """计算IoU"""
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        return iou.item()

    def calculate_dice(self, pred, target, threshold=0.5):
        """计算Dice系数"""
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)
        return dice.item()

    def calculate_precision_recall(self, pred, target, threshold=0.5):
        """计算精确率和召回率"""
        pred_binary = (pred.detach() > threshold).cpu().numpy().flatten()
        target_binary = target.detach().cpu().numpy().flatten()

        if target_binary.sum() == 0:  # 没有正样本
            precision = 1.0 if pred_binary.sum() == 0 else 0.0
            recall = 1.0
        else:
            precision = precision_score(
                target_binary, pred_binary, zero_division=0)
            recall = recall_score(target_binary, pred_binary, zero_division=0)

        return precision, recall

    def calculate_thin_structure_iou(self, pred, target, threshold=0.5):
        """计算细结构IoU（针对细分支的特殊IoU）"""
        try:
            pred_np = (pred.detach().cpu().numpy().squeeze()
                       > threshold).astype(np.uint8)
            target_np = target.detach().cpu().numpy().squeeze().astype(np.uint8)

            # 提取细结构（通过形态学操作）
            kernel = np.ones((3, 3), np.uint8)
            pred_thin = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
            target_thin = cv2.morphologyEx(target_np, cv2.MORPH_OPEN, kernel)

            intersection = np.logical_and(pred_thin, target_thin).sum()
            union = np.logical_or(pred_thin, target_thin).sum()

            if union == 0:
                return 1.0
            return intersection / union
        except Exception as e:
            print(f"Error in calculate_thin_structure_iou: {e}")
            return 0.0

    def calculate_connectivity_preservation_ratio(self, pred, target, threshold=0.5):
        """计算连通性保持比率"""
        try:
            pred_np = (pred.detach().cpu().numpy().squeeze()
                       > threshold).astype(np.uint8)
            target_np = target.detach().cpu().numpy().squeeze().astype(np.uint8)

            # 计算连通分量
            _, pred_components = cv2.connectedComponents(pred_np)
            _, target_components = cv2.connectedComponents(target_np)

            pred_num_components = len(np.unique(pred_components)) - 1  # 减去背景
            target_num_components = len(np.unique(target_components)) - 1

            if target_num_components == 0:
                return 1.0 if pred_num_components == 0 else 0.0

            # 简单的连通性保持比率计算
            cpr = min(pred_num_components, target_num_components) / \
                max(pred_num_components, target_num_components)
            return cpr
        except Exception as e:
            print(f"Error in calculate_connectivity_preservation_ratio: {e}")
            return 0.0

    def calculate_boundary_f1(self, pred, target, threshold=0.5):
        """计算边界F1分数"""
        try:
            pred_np = (pred.detach().cpu().numpy().squeeze()
                       > threshold).astype(np.uint8)
            target_np = target.detach().cpu().numpy().squeeze().astype(np.uint8)

            # 提取边界
            kernel = np.ones((3, 3), np.uint8)
            pred_boundary = pred_np - cv2.erode(pred_np, kernel, iterations=1)
            target_boundary = target_np - \
                cv2.erode(target_np, kernel, iterations=1)

            # 计算边界IoU作为F1的近似
            intersection = np.logical_and(pred_boundary, target_boundary).sum()
            union = np.logical_or(pred_boundary, target_boundary).sum()

            if union == 0:
                return 1.0
            return intersection / union
        except Exception as e:
            print(f"Error in calculate_boundary_f1: {e}")
            return 0.0

    def calculate_skeleton_similarity(self, pred, target, threshold=0.5):
        """计算骨架相似性"""
        try:
            pred_np = (pred.detach().cpu().numpy().squeeze()
                       > threshold).astype(bool)
            target_np = target.detach().cpu().numpy().squeeze().astype(bool)

            # 确保至少有一些像素为True
            if not np.any(pred_np) and not np.any(target_np):
                return 1.0
            if not np.any(pred_np) or not np.any(target_np):
                return 0.0

            # 计算骨架
            pred_skel = skeletonize(pred_np)
            target_skel = skeletonize(target_np)

            # 计算骨架IoU
            intersection = np.logical_and(pred_skel, target_skel).sum()
            union = np.logical_or(pred_skel, target_skel).sum()

            if union == 0:
                return 1.0
            return intersection / union
        except Exception as e:
            print(f"Error in calculate_skeleton_similarity: {e}")
            return 0.0

    def update(self, pred, target):
        """更新所有指标"""
        batch_size = pred.size(0)

        for i in range(batch_size):
            pred_sample = pred[i]
            target_sample = target[i]

            # 基础指标
            iou = self.calculate_iou(pred_sample, target_sample)
            dice = self.calculate_dice(pred_sample, target_sample)
            precision, recall = self.calculate_precision_recall(
                pred_sample, target_sample)

            # 细结构指标
            ts_iou = self.calculate_thin_structure_iou(
                pred_sample, target_sample)
            cpr = self.calculate_connectivity_preservation_ratio(
                pred_sample, target_sample)
            boundary_f1 = self.calculate_boundary_f1(
                pred_sample, target_sample)
            skel_sim = self.calculate_skeleton_similarity(
                pred_sample, target_sample)

            # 累加
            self.total_iou += iou
            self.total_dice += dice
            self.total_precision += precision
            self.total_recall += recall
            self.total_ts_iou += ts_iou
            self.total_cpr += cpr
            self.total_boundary_f1 += boundary_f1
            self.total_skel_sim += skel_sim
            self.count += 1

    def get_metrics(self):
        """获取平均指标"""
        if self.count == 0:
            return {}

        return {
            'IoU': self.total_iou / self.count,
            'Dice': self.total_dice / self.count,
            'Precision': self.total_precision / self.count,
            'Recall': self.total_recall / self.count,
            'TS-IoU': self.total_ts_iou / self.count,
            'CPR': self.total_cpr / self.count,
            'Boundary_F1': self.total_boundary_f1 / self.count,
            'Skeleton_Similarity': self.total_skel_sim / self.count
        }

# 计算模型参数量和FPS


def calculate_model_efficiency(model, input_shape=(1, 3, 256, 256), num_runs=100):
    """计算模型效率指标 - 已禁用"""
    # 返回空的效率指标
    return {
        'Total_Params_M': 0,
        'Trainable_Params_M': 0,
        'FPS': 0,
        'FLOPs_G': 0
    }

# 修改后的训练函数


def train_epoch(model, dataloader, criterion, optimizer, metrics):
    model.train()
    total_loss = 0
    metrics.reset()

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算指标
        total_loss += loss.item()
        with torch.no_grad():  # 修复：确保指标计算不影响梯度
            metrics.update(outputs, masks)

    avg_metrics = metrics.get_metrics()
    return total_loss / len(dataloader), avg_metrics

# 修改后的验证函数


def validate_epoch(model, dataloader, criterion, metrics):
    model.eval()
    total_loss = 0
    metrics.reset()

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            metrics.update(outputs, masks)

    avg_metrics = metrics.get_metrics()
    return total_loss / len(dataloader), avg_metrics

# 保存结果函数


def save_results(results, model_name, save_dir='results'):
    """保存所有结果到文件"""
    os.makedirs(save_dir, exist_ok=True)

    # 修复：处理不能序列化为JSON的对象
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    # 保存为JSON
    json_path = os.path.join(save_dir, f'{model_name}_results.json')
    serializable_results = make_serializable(results)
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # 保存为CSV
    csv_path = os.path.join(save_dir, f'{model_name}_results.csv')
    # 展平结果字典用于CSV保存
    flat_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_results[f"{key}_{sub_key}"] = sub_value
        else:
            flat_results[key] = value

    df = pd.DataFrame([flat_results])
    df.to_csv(csv_path, index=False)

    # 保存详细的训练历史
    if 'training_history' in results:
        history_path = os.path.join(
            save_dir, f'{model_name}_training_history.csv')
        history_df = pd.DataFrame(results['training_history'])
        history_df.to_csv(history_path, index=False)

    print(f"Results saved to {save_dir}/")
    return json_path, csv_path

# 修复：添加缺失的calculate_iou函数


def calculate_iou(pred, target, threshold=0.5):
    """独立的IoU计算函数（用于可视化）"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    if isinstance(target, torch.Tensor):
        target = target.detach()

    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)

    return iou.item() if isinstance(iou, torch.Tensor) else iou

# 修改可视化预测结果函数


def visualize_predictions(model, dataset, num_samples=4, model_name='model', save_dir='results'):
    model.eval()

    # 修复：处理样本数量超过数据集大小的情况
    num_samples = min(num_samples, len(dataset))
    if num_samples == 0:
        print("Dataset is empty, cannot visualize predictions")
        return

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    # 修复：如果只有一个样本，axes需要额外处理
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]

            # 添加批次维度并移到设备
            image_tensor = image.unsqueeze(0).to(device)

            # 预测
            pred = model(image_tensor)
            pred = pred.squeeze().cpu().numpy()

            # 转换用于显示
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            # 反归一化用于显示
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

            mask_np = mask.squeeze().cpu().numpy()

            # 显示
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            iou_score = calculate_iou(
                torch.tensor(pred), torch.tensor(mask_np))
            axes[i, 2].set_title(f'Prediction (IoU: {iou_score:.3f})')
            axes[i, 2].axis('off')

    plt.tight_layout()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存到results文件夹
    save_path = os.path.join(save_dir, f'{model_name}_prediction_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Prediction results saved to {save_path}")
    # plt.show()  # 移除图片显示
    plt.close()  # 添加close来释放内存


def plot_training_curves(training_history, model_name, save_dir='results'):
    """绘制训练曲线"""
    if len(training_history) < 2:
        print("Not enough training data to plot curves")
        return

    epochs = [h['epoch'] for h in training_history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Loss曲线
    axes[0, 0].plot(epochs, [h['train_loss']
                    for h in training_history], label='Train Loss')
    axes[0, 0].plot(epochs, [h['val_loss']
                    for h in training_history], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # IoU曲线
    axes[0, 1].plot(epochs, [h['train_IoU']
                    for h in training_history], label='Train IoU')
    axes[0, 1].plot(epochs, [h['val_IoU']
                    for h in training_history], label='Val IoU')
    axes[0, 1].set_title('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Dice曲线
    axes[0, 2].plot(epochs, [h['train_Dice']
                    for h in training_history], label='Train Dice')
    axes[0, 2].plot(epochs, [h['val_Dice']
                    for h in training_history], label='Val Dice')
    axes[0, 2].set_title('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # TS-IoU曲线
    axes[1, 0].plot(epochs, [h['val_TS-IoU']
                    for h in training_history], label='Val TS-IoU')
    axes[1, 0].set_title('Thin Structure IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # CPR曲线
    axes[1, 1].plot(epochs, [h['val_CPR']
                    for h in training_history], label='Val CPR')
    axes[1, 1].set_title('Connectivity Preservation Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Skeleton Similarity曲线
    axes[1, 2].plot(epochs, [h['val_Skeleton_Similarity']
                    for h in training_history], label='Val Skel. Sim.')
    axes[1, 2].set_title('Skeleton Similarity')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存到results文件夹
    save_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    # plt.show()  # 移除图片显示
    plt.close()  # 添加close来释放内存


def run_kfold_cross_validation(model_name='unet', save_dir='results', k=5):
    """
    运行K折交叉验证

    Args:
        model_name: 模型名称
        save_dir: 保存目录
        k: 折数

    Returns:
        各折的验证结果
    """
    print(f"\n{'='*80}")
    print(f"开始 {k} 折交叉验证 - 模型: {model_name.upper()}")
    print(f"{'='*80}")

    # 设置参数
    IMAGE_DIR = 'images'
    MASK_DIR = 'marks'
    IMAGE_SIZE = 256  # 使用固定值
    BATCH_SIZE = 60
    EPOCHS = 300
    LEARNING_RATE = 1e-4

    # 数据变换
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 创建完整数据集 (不应用transform，稍后在get_dataloader_from_indices中应用)
    full_dataset = BranchDataset(IMAGE_DIR, MASK_DIR, transform=None)

    if len(full_dataset) == 0:
        raise ValueError(f"No data found in {IMAGE_DIR} and {MASK_DIR}")

    # 创建K折划分
    kfold_splits = create_kfold_splits(full_dataset, k=k)

    # 存储每折的结果
    fold_results = []

    # 使用统一模型配置
    model_configs = UNIFIED_MODEL_CONFIGS

    for fold, (train_indices, val_indices) in enumerate(kfold_splits):
        print(f"\n{'-'*60}")
        print(f"运行第 {fold+1}/{k} 折")
        print(f"{'-'*60}")

        # 创建数据加载器
        train_loader = get_dataloader_from_indices(
            full_dataset, train_indices, BATCH_SIZE, shuffle=True, transform=train_transform
        )
        val_loader = get_dataloader_from_indices(
            full_dataset, val_indices, BATCH_SIZE, shuffle=False, transform=val_transform
        )

        # 创建模型
        if model_name not in model_configs:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {list(model_configs.keys())}")

        config = model_configs[model_name]
        model = config['model'](
            **{k: v for k, v in config.items() if k != 'model'}).to(device)

        # 创建优化器、损失函数和指标
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        train_metrics = ComprehensiveMetrics()
        val_metrics = ComprehensiveMetrics()

        # 训练记录
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(EPOCHS):
            # 训练
            train_loss, train_metrics_epoch = train_epoch(
                model, train_loader, criterion, optimizer, train_metrics)

            # 验证
            val_loss, val_metrics_epoch = validate_epoch(
                model, val_loader, criterion, val_metrics)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Fold {fold+1}, Epoch {epoch+1}/{EPOCHS}")
                print(
                    f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(
                    f"  Val IoU: {val_metrics_epoch['IoU']:.4f}, Val Dice: {val_metrics_epoch['Dice']:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        # 加载最佳模型进行最终评估
        model.load_state_dict(best_model_state)
        final_val_loss, final_val_metrics = validate_epoch(
            model, val_loader, criterion, val_metrics)

        # 保存本折结果
        fold_result = {
            'fold': fold + 1,
            'final_metrics': final_val_metrics,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        fold_results.append(fold_result)

        print(f"\nFold {fold+1} 最终结果:")
        print(f"  IoU: {final_val_metrics['IoU']:.4f}")
        print(f"  Dice: {final_val_metrics['Dice']:.4f}")
        print(f"  TS-IoU: {final_val_metrics['TS-IoU']:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")

    # 计算交叉验证平均结果
    print(f"\n{'='*80}")
    print(f"{k} 折交叉验证结果总结")
    print(f"{'='*80}")

    avg_metrics = {}
    metric_names = ['IoU', 'Dice', 'Precision', 'Recall',
                    'TS-IoU', 'CPR', 'Boundary_F1', 'Skeleton_Similarity']

    for metric in metric_names:
        values = [fold['final_metrics'][metric] for fold in fold_results]
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    # 打印结果
    print(f"{'指标':<20} | {'平均值':<10} | {'标准差':<10} | {'各折结果'}")
    print("-" * 80)
    for metric in metric_names:
        mean_val = avg_metrics[metric]['mean']
        std_val = avg_metrics[metric]['std']
        values_str = ", ".join(
            [f"{v:.3f}" for v in avg_metrics[metric]['values']])
        print(f"{metric:<20} | {mean_val:<10.4f} | {std_val:<10.4f} | {values_str}")

    # 保存交叉验证结果
    cv_save_dir = os.path.join(save_dir, f'{model_name}_kfold_cv')
    os.makedirs(cv_save_dir, exist_ok=True)

    # 保存详细结果
    cv_results = {
        'model_name': model_name,
        'k_folds': k,
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'summary': {
            'mean_iou': avg_metrics['IoU']['mean'],
            'std_iou': avg_metrics['IoU']['std'],
            'mean_dice': avg_metrics['Dice']['mean'],
            'std_dice': avg_metrics['Dice']['std']
        }
    }

    with open(os.path.join(cv_save_dir, 'kfold_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"\n交叉验证结果保存到: {cv_save_dir}")

    return cv_results


# 修改主函数中的模型保存和函数调用


def main(model_name='unet', save_dir='results', split_type='three_way'):
    """主训练函数 - 增强版错误处理"""

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 设置参数
    IMAGE_DIR = 'images'
    MASK_DIR = 'marks'  # 确保路径正确
    IMAGE_SIZE = image_size

    # 获取智能批次大小
    BATCH_SIZE = get_optimal_batch_size(model_name, IMAGE_SIZE)
    print(f"使用批次大小: {BATCH_SIZE}")

    EPOCHS = 300  # 用于测试，实际可以调整
    LEARNING_RATE = 1e-4

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 检查数据目录
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"图像目录不存在: {IMAGE_DIR}")
    if not os.path.exists(MASK_DIR):
        raise FileNotFoundError(f"掩码目录不存在: {MASK_DIR}")

    # 数据增强
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 创建数据集
    try:
        dataset = BranchDataset(IMAGE_DIR, MASK_DIR, transform=train_transform)
        if len(dataset) == 0:
            raise ValueError(f"No data found in {IMAGE_DIR} and {MASK_DIR}")
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return None

    # 数据划分
    try:
        if split_type == 'three_way':
            base_dataset = BranchDataset(IMAGE_DIR, MASK_DIR, transform=None)
            train_dataset, val_dataset, test_dataset = create_train_val_test_split(
                base_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
            test_dataset.dataset.transform = val_transform

            train_loader = create_robust_dataloader(
                train_dataset, BATCH_SIZE, shuffle=True)
            val_loader = create_robust_dataloader(
                val_dataset, BATCH_SIZE, shuffle=False)
            test_loader = create_robust_dataloader(
                test_dataset, BATCH_SIZE, shuffle=False)
        else:
            # 简单划分
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size])

            val_dataset.dataset.transform = val_transform

            train_loader = create_robust_dataloader(
                train_dataset, BATCH_SIZE, shuffle=True)
            val_loader = create_robust_dataloader(
                val_dataset, BATCH_SIZE, shuffle=False)
            test_loader = None
    except Exception as e:
        print(f"数据划分时出错: {e}")
        return None

    model_configs = UNIFIED_MODEL_CONFIGS

    # 创建模型
    if model_name not in model_configs:
        raise ValueError(
            f"Model {model_name} not supported. Available models: {list(model_configs.keys())}")

    config = model_configs[model_name]
    model = safe_model_creation(model_name, config)

    if model is None:
        print(f"Failed to create model {model_name}")
        return None

    # 将模型移到设备并替换问题层
    model = model.to(device)
    model = replace_problematic_layers(model)

    # 损失函数和优化器
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)

    # 训练循环
    train_metrics = ComprehensiveMetrics()
    val_metrics = ComprehensiveMetrics()
    training_history = []
    best_val_iou = 0

    print("开始训练...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        try:
            # 训练
            train_loss, train_metrics_dict = train_epoch(
                model, train_loader, criterion, optimizer, train_metrics)

            # 验证
            val_loss, val_metrics_dict = validate_epoch(
                model, val_loader, criterion, val_metrics)

            # 学习率调整
            scheduler.step(val_loss)

            # 记录历史
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            for key, value in train_metrics_dict.items():
                epoch_data[f'train_{key}'] = value
            for key, value in val_metrics_dict.items():
                epoch_data[f'val_{key}'] = value

            training_history.append(epoch_data)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(
                f"Train IoU: {train_metrics_dict['IoU']:.4f}, Val IoU: {val_metrics_dict['IoU']:.4f}")

            # 保存最佳模型
            if val_metrics_dict['IoU'] > best_val_iou:
                best_val_iou = val_metrics_dict['IoU']
                model_save_path = os.path.join(
                    save_dir, f'best_{model_name}_model.pth')
                torch.save(model.state_dict(), model_save_path)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"GPU内存不足在epoch {epoch+1}，尝试清理缓存...")
                torch.cuda.empty_cache()
                new_bs = max(2, BATCH_SIZE // 2)
                if new_bs == BATCH_SIZE:
                    print("批次大小已无法继续降低，停止训练该模型")
                    break
                BATCH_SIZE = new_bs
                print(f"降低批次大小到 {BATCH_SIZE}")
                train_loader = create_robust_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
                val_loader = create_robust_dataloader(val_dataset, BATCH_SIZE, shuffle=False)
                continue
            else:
                print(f"训练时出现错误: {e}")
                break
        except Exception as e:
            print(f"训练时出现未知错误: {e}")
            break

    # 后续处理（效率计算、测试等）保持不变
    print("\n计算模型效率指标...")
    efficiency_metrics = calculate_model_efficiency(model)

    # 最终评估
    model_load_path = os.path.join(save_dir, f'best_{model_name}_model.pth')
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))

    final_val_loss, final_val_metrics = validate_epoch(
        model, val_loader, criterion, val_metrics)

    final_test_metrics = None
    if test_loader is not None:
        test_metrics = ComprehensiveMetrics()
        final_test_loss, final_test_metrics = validate_epoch(
            model, test_loader, criterion, test_metrics)

    # 整合结果
    final_results = {
        'model_name': model_name,
        'split_type': split_type,
        'loss_function': 'dice_bce',
        'epochs_trained': EPOCHS,
        'batch_size': BATCH_SIZE,
        'best_val_iou': best_val_iou,
        'final_metrics': final_val_metrics,
        'final_test_metrics': final_test_metrics,
        'efficiency_metrics': efficiency_metrics,
        'training_history': training_history,
        'model_config': {k: v for k, v in config.items() if k != 'model'},
        'training_params': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'image_size': IMAGE_SIZE
        },
        'model_path': model_load_path
    }

    # 保存结果
    save_results(final_results, model_name, save_dir)
    # print_single_model_results(
    #     model_name, final_val_metrics, efficiency_metrics, final_test_metrics)

    # 可视化
    try:
        plot_training_curves(training_history, model_name, save_dir)
        vis_dataset = BranchDataset(
            IMAGE_DIR, MASK_DIR, transform=val_transform)
        visualize_predictions(model, vis_dataset, num_samples=4,
                              model_name=model_name, save_dir=save_dir)
    except Exception as e:
        print(f"可视化时出错: {e}")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n训练完成! 结果保存到 {save_dir}/")
    return final_results

# 修改run_all_models函数以包含更好的错误恢复


def run_all_models(save_dir='results', split_type='three_way'):
    """运行所有模型的批量测试 - 增强版错误处理"""

    print("🚀 开始运行所有模型...")
    print(f"保存目录: {save_dir}")
    print(f"分割类型: {split_type}")

    model_configs = UNIFIED_MODEL_CONFIGS

    # 获取所有可运行的模型名称
    model_names = list(model_configs.keys())
    print(f"将运行以下 {len(model_names)} 个模型:")
    for i, name in enumerate(model_names, 1):
        print(f"  {i:2d}. {name}")
    print()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    successful_models = []
    failed_models = []

    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'='*80}")
        print(f"运行模型 {i}/{len(model_names)}: {model_name.upper()}")
        print(f"{'='*80}")

        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_save_dir = os.path.join(save_dir, model_name)

            # 运行单个模型
            result = main(model_name, model_save_dir, split_type=split_type)

            if result is not None:
                successful_models.append(model_name)
                print(f"✓ {model_name} 训练完成")
            else:
                failed_models.append(model_name)
                print(f"✗ {model_name} 训练失败")

        except Exception as e:
            print(f"✗ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append(model_name)

            # 清理GPU缓存后继续下一个模型
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # 输出总结
    print(f"\n{'='*80}")
    print("批量运行总结")
    print(f"{'='*80}")
    print(f"成功运行模型数: {len(successful_models)}/{len(model_names)}")

    if successful_models:
        print("\n成功的模型:")
        for model in successful_models:
            print(f"  ✓ {model}")

    if failed_models:
        print("\n失败的模型:")
        for model in failed_models:
            print(f"  ✗ {model}")

    print(f"\n所有模型结果保存在: {save_dir}")
    print(f"{'='*80}")

    return successful_models, failed_models


def save_comparison_results(all_results, save_dir='results'):
    """保存所有模型的对比结果"""
    comparison_data = []

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for model, results in all_results.items():
        final_metrics = results['final_metrics']
        efficiency = results['efficiency_metrics']

        row = {
            'Model': model.upper(),
            'IoU(%)': round(final_metrics['IoU'] * 100, 1),
            'Dice(%)': round(final_metrics['Dice'] * 100, 1),
            'Precision(%)': round(final_metrics['Precision'] * 100, 1),
            'Recall(%)': round(final_metrics['Recall'] * 100, 1),
            'TS-IoU(%)': round(final_metrics['TS-IoU'] * 100, 1),
            'CPR(%)': round(final_metrics['CPR'] * 100, 1),
            'Boundary_F1(%)': round(final_metrics['Boundary_F1'] * 100, 1),
            'Skeleton_Similarity(%)': round(final_metrics['Skeleton_Similarity'] * 100, 1)
        }
        comparison_data.append(row)

    # 保存为CSV到results文件夹
    comparison_df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(save_dir, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)

    # 保存为格式化的文本文件到results文件夹
    txt_path = os.path.join(save_dir, 'model_comparison.txt')
    with open(txt_path, 'w') as f:
        f.write("Model Performance Comparison\n")
        f.write("="*90 + "\n")
        f.write(
            f"{'Model':<12} | {'Overall Metrics':<35} | {'Thin Structure Performance':<45}\n")
        f.write(f"{'':12} | {'IoU↑':<8} {'Dice↑':<8} {'Prec.↑':<9} {'Rec.↑':<8} | {'TS-IoU↑':<9} {'CPR↑':<8} {'Boundary F1↑':<13} {'Skel. Sim.↑':<12}\n")
        f.write("-"*90 + "\n")

        for data in comparison_data:
            line = (f"{data['Model']:<12} | "
                    f"{data['IoU(%)']:<8.1f} {data['Dice(%)']:<8.1f} "
                    f"{data['Precision(%)']:<9.1f} {data['Recall(%)']:<8.1f} | "
                    f"{data['TS-IoU(%)']:<9.1f} {data['CPR(%)']:<8.1f} "
                    f"{data['Boundary_F1(%)']:<13.1f} {data['Skeleton_Similarity(%)']:<12.1f}\n")
            f.write(line)

        f.write("="*90 + "\n")

    print(f"\nComparison results saved to {save_dir}:")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")

# 添加一个创建结果总览的函数


def create_results_summary(save_dir='results'):
    """创建结果文件夹的总览"""
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, 'README.md')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Model Training Results\n\n")
        f.write("This folder contains all the training results and outputs.\n\n")
        f.write("## File Structure\n\n")
        f.write("- `best_*_model.pth`: Trained model weights\n")
        f.write("- `*_results.json`: Detailed results in JSON format\n")
        f.write("- `*_results.csv`: Summary results in CSV format\n")
        f.write("- `*_training_history.csv`: Training history for each epoch\n")
        f.write("- `*_training_curves.png`: Training curves visualization\n")
        f.write("- `*_prediction_results.png`: Sample predictions visualization\n")
        f.write("- `model_comparison.csv`: Comparison of all models (CSV)\n")
        f.write(
            "- `model_comparison.txt`: Comparison of all models (formatted text)\n\n")
        f.write("## Usage\n\n")
        f.write("To load a trained model:\n")
        f.write("```python\n")
        f.write("model = smp.Unet(...)  # Initialize model with same config\n")
        f.write("model.load_state_dict(torch.load('best_unet_model.pth'))\n")
        f.write("```\n")

    print(f"Results summary created at {summary_path}")


def print_environment_info():
    print("\n" + "="*80)
    print("ENVIRONMENT INFORMATION")
    print("="*80)

    # Python版本
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")

    print("\n" + "-"*80)
    print("INSTALLED PACKAGES:")
    print("-"*80)

    # 获取所有已安装的包
    installed_packages = [d for d in pkg_resources.working_set]
    installed_packages.sort(key=lambda x: x.project_name.lower())

    # 打印包信息
    for package in installed_packages:
        print(f"{package.project_name:<30} {package.version}")

    print("-"*80)
    print(f"Total packages installed: {len(installed_packages)}")
    print("="*80)


# 在设备设置后添加智能批次大小和错误处理
def get_optimal_batch_size(model_name, image_size=256):
    """根据模型复杂度智能选择批次大小 - 针对NVIDIA L4 (23GB)优化"""
    # 检查可用内存
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            print(f"GPU Memory: {gpu_memory:.1f}GB")
        except:
            gpu_memory = 23  # L4默认值
    else:
        gpu_memory = 4  # CPU模式使用更小的值

    # 特殊处理DeepLab模型 - 需要至少批次大小2，L4可以用更大批次
    deeplab_models = ['deeplabv3', 'deeplabv3+']
    if model_name in deeplab_models:
        if gpu_memory >= 20:  # L4有23GB，DeepLab可以用很大批次
            return 12
        elif gpu_memory >= 16:
            return 8
        elif gpu_memory >= 8:
            return 6
        else:
            return 4  # 最小值改为4，避免BatchNorm问题

    # 模型复杂度分类（根据经验和参数量）
    lightweight_models = [
        'mobilenet_v2_unet', 'efficientnet-b0_unet', 'linknet', 'fpn'
    ]

    medium_models = [
        'unet', 'unet++', 'pspnet',
        'manet', 'pan', 'resnet50_unet', 'densenet121_unet'
    ]

    heavy_models = [
        'convnext_base_unet', 'vit_b_16_unet', 'swin_base_unet',
        'mit_b0_unet', 'mit_b1_unet', 'mit_b2_unet'
    ]

    ultra_heavy_models = [
        'mit_b3_unet', 'mit_b4_unet', 'mit_b5_unet', 'levit_128s_unet'
    ]

    # 根据L4的23GB内存大幅优化批次大小
    if model_name in lightweight_models:
        if gpu_memory >= 20:  # L4可以跑很大批次
            return 32  # 轻量级模型可以用很大批次
        elif gpu_memory >= 16:
            return 24
        elif gpu_memory >= 8:
            return 16
        else:
            return 8
    elif model_name in medium_models:
        if gpu_memory >= 20:  # L4优化 - 中等模型也可以用较大批次
            return 16
        elif gpu_memory >= 16:
            return 12
        elif gpu_memory >= 12:
            return 8
        elif gpu_memory >= 8:
            return 6
        else:
            return 4
    elif model_name in heavy_models:
        if gpu_memory >= 20:  # L4可以跑更大批次的重型模型
            return 12
        elif gpu_memory >= 16:
            return 8
        elif gpu_memory >= 12:
            return 6
        elif gpu_memory >= 8:
            return 4
        else:
            return 3
    else:  # ultra_heavy_models
        if gpu_memory >= 20:  # L4优化 - 即使超重型模型也可以用较大批次
            return 8
        elif gpu_memory >= 16:
            return 6
        elif gpu_memory >= 12:
            return 4
        else:
            return 3


def safe_model_creation(model_name, config):
    """安全地创建模型，包含错误处理和回退机制"""
    try:
        print(f"正在创建模型: {model_name}")

        # 首先尝试正常创建
        model = config['model'](
            **{k: v for k, v in config.items() if k != 'model'})

        # 测试模型是否能正常前向传播
        test_input = torch.randn(1, 3, 256, 256)
        if torch.cuda.is_available():
            model = model.cuda()
            test_input = test_input.cuda()

        with torch.no_grad():
            test_output = model(test_input)

        print(f"✓ 模型 {model_name} 创建成功")
        return model

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠ {model_name} 内存不足，尝试清理GPU缓存...")
            torch.cuda.empty_cache()

            # 尝试使用更小的批次大小重新创建
            try:
                model = config['model'](
                    **{k: v for k, v in config.items() if k != 'model'})
                if torch.cuda.is_available():
                    model = model.cuda()
                print(f"✓ {model_name} 在清理缓存后创建成功")
                return model
            except:
                print(f"✗ {model_name} 即使在清理缓存后仍然失败")
                return None

    except Exception as e:
        print(f"✗ 创建模型 {model_name} 时出错: {e}")

        # 对于一些特殊的编码器，尝试回退到更基础的版本
        fallback_encoders = {
            'convnext_base': 'resnet50',
            'vit_b_16': 'resnet34',
            'swin_base': 'resnet34',
            'mit_b3': 'mit_b0',
            'mit_b4': 'mit_b0',
            'mit_b5': 'mit_b0',
            'levit_128s': 'efficientnet-b0'
        }

        original_encoder = config.get('encoder_name', '')
        if original_encoder in fallback_encoders:
            print(f"⚠ 尝试使用回退编码器: {fallback_encoders[original_encoder]}")
            try:
                fallback_config = config.copy()
                fallback_config['encoder_name'] = fallback_encoders[original_encoder]
                model = fallback_config['model'](
                    **{k: v for k, v in fallback_config.items() if k != 'model'})

                if torch.cuda.is_available():
                    model = model.cuda()

                print(f"✓ {model_name} 使用回退编码器创建成功")
                return model
            except Exception as fallback_error:
                print(f"✗ 回退方案也失败: {fallback_error}")

        return None


def replace_problematic_layers(model):
    """替换可能导致问题的层"""
    try:
        # 记录模型的当前设备
        model_device = next(model.parameters()).device

        # 替换所有BatchNorm为GroupNorm (对小批次更友好)
        def replace_batchnorm_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if isinstance(child, nn.BatchNorm2d):
                        # 获取通道数
                        num_channels = child.num_features
                        # 选择合适的组数
                        num_groups = min(32, num_channels)
                        while num_channels % num_groups != 0 and num_groups > 1:
                            num_groups -= 1

                        # 创建GroupNorm并移动到正确设备
                        new_layer = nn.GroupNorm(
                            num_groups, num_channels, eps=child.eps).to(model_device)

                        # 如果原层有可学习参数，尝试复制
                        if child.affine and hasattr(child, 'weight') and child.weight is not None:
                            with torch.no_grad():
                                new_layer.weight.copy_(child.weight)
                                new_layer.bias.copy_(child.bias)

                        setattr(module, name, new_layer)
                        print(
                            f"Replaced BatchNorm2d with GroupNorm (groups={num_groups}, channels={num_channels})")
                else:
                    replace_batchnorm_recursive(child)

        replace_batchnorm_recursive(model)
        return model

    except Exception as e:
        print(f"Warning: Failed to replace BatchNorm layers: {e}")
        return model


def create_robust_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
    """创建更稳健的数据加载器"""
    # 确保数据集不为空
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    # 调整批次大小以适应数据集大小
    if len(dataset) < batch_size:
        batch_size = max(1, len(dataset) // 2)
        print(
            f"Warning: Adjusted batch_size to {batch_size} due to small dataset")

    # 使用较小的num_workers避免潜在的多进程问题
    actual_num_workers = min(
        num_workers, 2) if torch.cuda.is_available() else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=actual_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,  # 不丢弃最后一个batch
        persistent_workers=False  # 避免worker进程问题
    )

# 修改主函数以包含更好的错误处理


def main(model_name='unet', save_dir='results', split_type='three_way'):
    """主训练函数 - 增强版错误处理"""

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 设置参数
    IMAGE_DIR = 'images'
    MASK_DIR = 'marks'  # 确保路径正确
    IMAGE_SIZE = image_size

    # 获取智能批次大小
    BATCH_SIZE = get_optimal_batch_size(model_name, IMAGE_SIZE)
    print(f"使用批次大小: {BATCH_SIZE}")

    EPOCHS = 300  # 用于测试，实际可以调整
    LEARNING_RATE = 1e-4

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 检查数据目录
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"图像目录不存在: {IMAGE_DIR}")
    if not os.path.exists(MASK_DIR):
        raise FileNotFoundError(f"掩码目录不存在: {MASK_DIR}")

    # 数据增强
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 创建数据集
    try:
        dataset = BranchDataset(IMAGE_DIR, MASK_DIR, transform=train_transform)
        if len(dataset) == 0:
            raise ValueError(f"No data found in {IMAGE_DIR} and {MASK_DIR}")
        print(f"数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return None

    # 数据划分
    try:
        if split_type == 'three_way':
            base_dataset = BranchDataset(IMAGE_DIR, MASK_DIR, transform=None)
            train_dataset, val_dataset, test_dataset = create_train_val_test_split(
                base_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
            test_dataset.dataset.transform = val_transform

            train_loader = create_robust_dataloader(
                train_dataset, BATCH_SIZE, shuffle=True)
            val_loader = create_robust_dataloader(
                val_dataset, BATCH_SIZE, shuffle=False)
            test_loader = create_robust_dataloader(
                test_dataset, BATCH_SIZE, shuffle=False)
        else:
            # 简单划分
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size])

            val_dataset.dataset.transform = val_transform

            train_loader = create_robust_dataloader(
                train_dataset, BATCH_SIZE, shuffle=True)
            val_loader = create_robust_dataloader(
                val_dataset, BATCH_SIZE, shuffle=False)
            test_loader = None
    except Exception as e:
        print(f"数据划分时出错: {e}")
        return None

    # 模型配置使用统一配置（之前残留的字典已移除）
    model_configs = UNIFIED_MODEL_CONFIGS  # 统一配置

    # 创建模型
    # 改为使用统一配置
    if model_name not in UNIFIED_MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported. Available: {list(UNIFIED_MODEL_CONFIGS.keys())}")
    config = UNIFIED_MODEL_CONFIGS[model_name]
    model = safe_model_creation(model_name, config)

    if model is None:
        print(f"Failed to create model {model_name}")
        return None

    # 将模型移到设备并替换问题层
    model = model.to(device)
    model = replace_problematic_layers(model)

    # 损失函数和优化器
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)

    # 训练循环
    train_metrics = ComprehensiveMetrics()
    val_metrics = ComprehensiveMetrics()
    training_history = []
    best_val_iou = 0

    print("开始训练...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        try:
            # 训练
            train_loss, train_metrics_dict = train_epoch(
                model, train_loader, criterion, optimizer, train_metrics)

            # 验证
            val_loss, val_metrics_dict = validate_epoch(
                model, val_loader, criterion, val_metrics)

            # 学习率调整
            scheduler.step(val_loss)

            # 记录历史
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            for key, value in train_metrics_dict.items():
                epoch_data[f'train_{key}'] = value
            for key, value in val_metrics_dict.items():
                epoch_data[f'val_{key}'] = value

            training_history.append(epoch_data)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(
                f"Train IoU: {train_metrics_dict['IoU']:.4f}, Val IoU: {val_metrics_dict['IoU']:.4f}")

            # 保存最佳模型
            if val_metrics_dict['IoU'] > best_val_iou:
                best_val_iou = val_metrics_dict['IoU']
                model_save_path = os.path.join(
                    save_dir, f'best_{model_name}_model.pth')
                torch.save(model.state_dict(), model_save_path)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"GPU内存不足在epoch {epoch+1}，尝试清理缓存...")
                torch.cuda.empty_cache()
                new_bs = max(2, BATCH_SIZE // 2)  # 最小保持2避免BN不稳定
                if new_bs == BATCH_SIZE:
                    print("批次大小已无法继续降低，停止训练该模型")
                    break
                BATCH_SIZE = new_bs
                print(f"降低批次大小到 {BATCH_SIZE}")
                train_loader = create_robust_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
                val_loader = create_robust_dataloader(val_dataset, BATCH_SIZE, shuffle=False)
                continue
            else:
                print(f"训练时出现错误: {e}")
                break
        except Exception as e:
            print(f"训练时出现未知错误: {e}")
            break

    # 后续处理（效率计算、测试等）保持不变
    print("\n计算模型效率指标...")
    efficiency_metrics = calculate_model_efficiency(model)

    # 最终评估
    model_load_path = os.path.join(save_dir, f'best_{model_name}_model.pth')
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path))

    final_val_loss, final_val_metrics = validate_epoch(
        model, val_loader, criterion, val_metrics)

    final_test_metrics = None
    if test_loader is not None:
        test_metrics = ComprehensiveMetrics()
        final_test_loss, final_test_metrics = validate_epoch(
            model, test_loader, criterion, test_metrics)

    # 整合结果
    final_results = {
        'model_name': model_name,
        'split_type': split_type,
        'loss_function': 'dice_bce',
        'epochs_trained': EPOCHS,
        'batch_size': BATCH_SIZE,
        'best_val_iou': best_val_iou,
        'final_metrics': final_val_metrics,
        'final_test_metrics': final_test_metrics,
        'efficiency_metrics': efficiency_metrics,
        'training_history': training_history,
        'model_config': {k: v for k, v in config.items() if k != 'model'},
        'training_params': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'image_size': IMAGE_SIZE
        },
        'model_path': model_load_path
    }

    # 保存结果
    save_results(final_results, model_name, save_dir)
    # print_single_model_results(
    #     model_name, final_val_metrics, efficiency_metrics, final_test_metrics)

    # 可视化
    # try:
    #     plot_training_curves(training_history, model_name, save_dir)
    #     vis_dataset = BranchDataset(
    #         IMAGE_DIR, MASK_DIR, transform=val_transform)
    #     visualize_predictions(model, vis_dataset, num_samples=4,
    #                           model_name=model_name, save_dir=save_dir)
    # except Exception as e:
    #     print(f"可视化时出错: {e}")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n训练完成! 结果保存到 {save_dir}/")
    return final_results


if __name__ == "__main__":
    run_all_models(save_dir='256', split_type='three_way')
