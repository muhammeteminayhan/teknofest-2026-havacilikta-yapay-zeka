"""Görev 3 için minimal DINO omurgası (fork'suz).

`TemplateMatcher` omurgadan yalnızca şunları bekler: `backbone(tensor)` çağrısı
sonrası `.last_cls_token`, `.last_patch_tokens`, `.mask_dim` öznitelikleri ve
`.parameters()`. Fork'un ağır P3/P4/P5 projeksiyon katmanları eşleme için GEREKSİZ
(TemplateMatcher onları kullanmıyor), bu yüzden burada yalnızca ham token çıkarımı
yapan sade bir sarmalayıcı var.

İki omurga:
  - "dinov3": HuggingFace `facebook/dinov3-vits16-pretrain-lvd1689m` (patch16, gated;
     bir kez indirilip önbelleğe alınınca token'sız çevrimdışı çalışır). VARSAYILAN.
  - "dinov2": torch.hub `dinov2_vits14_reg` (patch14, gated değil; yedek).

İkisi de ImageNet normalize eder ve girişi patch boyutunun katına doldurur; token
çıkarım biçimi fork'un `dinov3.py` (satır 78-119) mantığıyla birebir.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _DinoV3Backbone(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m"):
        super().__init__()
        from transformers import AutoModel
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception:
            # ağ kapalıyken (yarışma) yalnızca yerel önbellekten yükle
            self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.patch_size = self.backbone.config.patch_size            # 16
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
            _, _, H, W = x.shape
        x = (x - self.mean) / self.std
        with torch.no_grad():
            out = self.backbone(pixel_values=x)
            lhs = out.last_hidden_state                  # (B, 1+R+N, D)
            n_special = 1 + self.num_register_tokens
            patch_tokens = lhs[:, n_special:, :]
            cls = lhs[:, 0, :]
        self.last_cls_token = cls
        self.last_patch_tokens = patch_tokens
        self.mask_dim = (H // self.patch_size, W // self.patch_size)
        return patch_tokens


class _DinoV2Backbone(nn.Module):
    """Yedek omurga (gated değil). torch.hub repo kodu önbellekte olmalı."""
    def __init__(self, model_name="dinov2_vits14_reg"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.patch_size = self.backbone.patch_size       # 14
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
            _, _, H, W = x.shape
        x = (x - self.mean) / self.std
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            patch_tokens = feats["x_norm_patchtokens"]
            cls = feats["x_norm_clstoken"]
        self.last_cls_token = cls
        self.last_patch_tokens = patch_tokens
        self.mask_dim = (H // self.patch_size, W // self.patch_size)
        return patch_tokens


def build_backbone(kind: str = "dinov3", device: str = None) -> nn.Module:
    """DINO omurgası kur ve cihaza taşı. kind: 'dinov3' (varsayılan) | 'dinov2'."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = _DinoV3Backbone() if kind == "dinov3" else _DinoV2Backbone()
    return backbone.to(device).eval()
