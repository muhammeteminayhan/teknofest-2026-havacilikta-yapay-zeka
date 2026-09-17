import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3(nn.Module):
    """
    DINOv3 (patch16, Meta AI, Aug 2025 release) backbone for YOLOv8.
    Implements Gram Anchoring and Multi-Scale Spatial Reshaping (P3, P4, P5).

    IMPORTANT: imgsz must be a multiple of 32 (e.g. 640, 672, 704...).
    At patch16, this guarantees P3/P4/P5 all land exactly on the native
    patch grid with no padding or interpolation required:
      - P4 (stride 16) == native patch grid, no resize.
      - P5 (stride 32) == native grid downsampled exactly 2x.
      - P3 (stride 8)  == native grid upsampled exactly 2x.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        p3_ch=256, p4_ch=512, p5_ch=1024,
        normalize_input=True,
    ):
        super().__init__()

        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.embed_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size  # 16
        self.num_register_tokens = getattr(self.backbone.config, 'num_register_tokens', 4)

        self.normalize_input = normalize_input
        self.register_buffer(
            "mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # Gram Anchoring MLP
        reduction = 16
        self.gram_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // reduction, 1)
        )

        # Multi-Scale Projection Layers
        from ultralytics.nn.modules.conv import Conv, ConvTranspose

        # P3: native grid upsampled exactly 2x (stride 8)
        self.p3_proj = Conv(self.embed_dim, p3_ch, 1)
        self.p3_upsample = ConvTranspose(p3_ch, p3_ch, 2, 2)

        # P4: native grid, no resize (stride 16)
        self.p4_proj = Conv(self.embed_dim, p4_ch, 1)

        # P5: native grid downsampled exactly 2x (stride 32)
        self.p5_proj = Conv(self.embed_dim, p5_ch, 1)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def train(self, mode=True):
        """Override train to ensure frozen backbone stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        B, C, H, W = x.shape

        # Auto-pad to a multiple of the native patch size so H/W divide the
        # patch grid exactly. With imgsz already a multiple of 32 (and 32 a
        # multiple of patch_size=16) this is normally a no-op; it only
        # engages as a safety net for odd input sizes.
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H, W = x.shape

        if self.normalize_input:
            x = (x - self.mean) / self.std

        # Forward pass through frozen DINOv3
        with torch.no_grad():
            if hasattr(self.backbone, 'forward_features'):
                # PyTorch Hub format (legacy checkpoint support)
                features = self.backbone.forward_features(x)
                patch_tokens = features['x_norm_patchtokens']
                last_cls_token = features.get('x_norm_clstoken', None)
            else:
                # Hugging Face format
                outputs = self.backbone(pixel_values=x)
                last_hidden_state = outputs.last_hidden_state  # (B, 1 + R + N, D)
                num_special_tokens = 1 + self.num_register_tokens  # CLS + register tokens
                patch_tokens = last_hidden_state[:, num_special_tokens:, :]  # (B, N, D)
                last_cls_token = last_hidden_state[:, 0, :]

        if torch.isnan(patch_tokens).any() or torch.isinf(patch_tokens).any():
            print("WARNING: NaN/Inf detected in DINO patch tokens! "
                  "Check input normalization / corrupt images upstream of DINOv3.")

        self.last_cls_token = last_cls_token
        self.last_patch_tokens = patch_tokens

        # Native patch grid — exact, no rounding, since H/W are multiples of 32
        h_patches, w_patches = H // self.patch_size, W // self.patch_size
        self.mask_dim = (h_patches, w_patches)

        # Reshape to 2D grid: (B, D, h_patches, w_patches)
        F_spatial = patch_tokens.reshape(
            B, h_patches, w_patches, self.embed_dim
        ).permute(0, 3, 1, 2).contiguous()

        # Gram Anchoring Operation — forced fp32, isolated from autocast
        N = h_patches * w_patches
        with torch.autocast(device_type='cuda', enabled=False):
            F_flat32 = F_spatial.float().view(B, self.embed_dim, N)
            G = torch.bmm(F_flat32, F_flat32.transpose(1, 2)) / N
            
            # Match the linear layer dtype (which might be FP16 during validation)
            G = G.to(self.gram_mlp[0].weight.dtype)
            
            alpha_raw = self.gram_mlp(G)
            alpha = torch.sigmoid(alpha_raw)

        alpha = alpha.to(F_spatial.dtype).view(B, self.embed_dim, 1, 1)
        F_anchored = F_spatial * alpha

        # Multi-Scale Projections — already exact, native strides (8, 16, 32);
        # no interpolation or cropping needed since h_patches/w_patches were
        # derived directly from self.patch_size above.
        p3 = self.p3_upsample(self.p3_proj(F_anchored))   # stride 8
        p4 = self.p4_proj(F_anchored)                      # stride 16, native
        p5 = self.p5_downsample(self.p5_proj(F_anchored))  # stride 32

        return [p3, p4, p5]
