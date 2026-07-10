"""
Template Matcher — DINOv2 FPN-style multi-scale matching
=========================================================
Uses Prakhar's DinoV2Patches backbone (shared with YOLO head).
FPN pyramid at 3 scales → per-scale cosine similarity heatmaps
→ max-fused heatmap → bbox extraction with morphological closing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage import zoom as nd_zoom, gaussian_filter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Detection:
    object_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float

    def to_xyxy(self):
        x1 = int(self.x_center - self.width / 2)
        y1 = int(self.y_center - self.height / 2)
        x2 = int(self.x_center + self.width / 2)
        y2 = int(self.y_center + self.height / 2)
        return x1, y1, x2, y2


@dataclass
class ReferenceObject:
    object_id: int
    cls_tokens: List[torch.Tensor] = field(default_factory=list)
    patch_tokens: List[torch.Tensor] = field(default_factory=list)
    mean_cls: Optional[torch.Tensor] = None
    mean_patches: Optional[torch.Tensor] = None


# ── Main class ─────────────────────────────────────────────────────────────────

class TemplateMatcher:
    """
    Zero-shot template matching using DINOv2 patch features.
    Plugs into YoloDino alongside the YOLO detection head.

    FPN levels (P3/P4/P5) run at different resolutions and are
    max-fused into a single heatmap for robust multi-scale detection.
    """

    # FPN pyramid will be loaded from config in __init__
    FPN_LEVELS = []

    def __init__(self, backbone, config: dict):
        self.backbone        = backbone
        self.config          = config
        
        self.cls_threshold   = config['matching']['cls_threshold']
        self.min_confidence  = config['matching']['min_confidence']
        self.threshold_ratio = config['matching']['threshold_ratio']
        self.noise_floor     = config['matching']['noise_floor']
        self.fusion          = config['fpn']['fusion_method']
        self.augment         = config['reference']['augment']
        
        # Load FPN levels from config
        self.FPN_LEVELS = [(k, v) for k, v in config['fpn']['levels'].items()]
        
        self.references: List[ReferenceObject] = []

        self.device = next(backbone.parameters()).device
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std  = (0.229, 0.224, 0.225)

    # ── Module 1: Preprocessing ────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray, size: int) -> torch.Tensor:
        """Resize image preserving aspect ratio, snap to multiple of 14, normalize."""
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h = max(14, (int(h * scale) // 14) * 14)
        new_w = max(14, (int(w * scale) // 14) * 14)
        
        # Use INTER_CUBIC for better feature preservation
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        tensor = transforms.ToTensor()(resized)
        # DinoV2Patches backbone will handle the ImageNet normalization internally
        
        return tensor.unsqueeze(0).to(self.device)


    def _augment(self, image: np.ndarray) -> List[np.ndarray]:
        """Augment reference with flips and scale jitter for robustness."""
        if not self.augment:
            return [image]
        h, w = image.shape[:2]
        augmented = [
            image,
            cv2.flip(image, 1),   # horizontal flip
            cv2.flip(image, 0),   # vertical flip
        ]
        for s in self.config['reference']['augment_scales']:
            scaled = cv2.resize(image, (int(w * s), int(h * s)))
            # center crop back to original size
            sh = min(scaled.shape[0], h)
            sw = min(scaled.shape[1], w)
            y0 = (scaled.shape[0] - sh) // 2
            x0 = (scaled.shape[1] - sw) // 2
            crop = scaled[y0:y0+sh, x0:x0+sw]
            augmented.append(cv2.resize(crop, (w, h)))
        return augmented

    # ── Module 2: Feature Extractor ────────────────────────────────────────────

    @torch.no_grad()
    def _extract(self, image: np.ndarray, size: int):
        """Run backbone at given size. Returns cls (1,C), patches (1,N,C), (gh, gw)."""
        print(f"Preprocessing image of size {image.shape} to target size {size}")
        tensor = self._preprocess(image, size)
        print("Running backbone forward...")
        self.backbone(tensor)
        print("Backbone forward complete!")
        cls     = self.backbone.last_cls_token.clone()    # (1, C)
        patches = self.backbone.last_patch_tokens.clone() # (1, N, C)
        gh, gw  = self.backbone.mask_dim
        return cls, patches, gh, gw

    # ── Module 3: Reference Registration ──────────────────────────────────────

    def add_reference(self, images: List[np.ndarray], object_id: int = None) -> int:
        """Register a reference object. Augments and caches features at P3 scale."""
        if object_id is None:
            object_id = len(self.references)

        ref = ReferenceObject(object_id=object_id)
        size = self.FPN_LEVELS[0][1]  # use finest scale for reference

        for img in images:
            for aug in self._augment(img):
                cls, patches, _, _ = self._extract(aug, size)
                ref.cls_tokens.append(cls)
                ref.patch_tokens.append(patches)

        ref.mean_cls     = torch.stack(ref.cls_tokens).mean(dim=0)    # (1, C)
        ref.mean_patches = torch.stack(ref.patch_tokens).mean(dim=0)  # (1, N, C)

        self.references.append(ref)
        print(f"Reference {object_id} registered — "
              f"{len(ref.cls_tokens)} augmented views, "
              f"patch shape: {ref.mean_patches.shape}")
        return object_id

    # ── Module 3: CLS Filter ──────────────────────────────────────────────────
    
    # Note: Global CLS filter removed. Comparing a full scene CLS token 
    # to an isolated template CLS token yields mathematically low similarities 
    # and causes false negatives. All references are now evaluated.

    # ── Module 4: FPN Heatmap ─────────────────────────────────────────────────

    @torch.no_grad()
    def _fpn_heatmap(
        self,
        H_orig: int,
        W_orig: int,
        ref: ReferenceObject,
        frame_feats: dict,
    ) -> np.ndarray:
        """
        Build multi-scale similarity heatmap for one reference object.
        Uses CLS token as template vector compared against each frame patch.
        Per-scale maps are upsampled to original resolution then fused.

        OPTIMIZASYON: kare oznitelikleri (`frame_feats`) match() icinde olcek
        basina BIR KEZ cikarilip burada tekrar kullanilir — boylece maliyet
        referans sayisindan bagimsizdir (eskiden her referans icin yeniden
        cikariliyordu). Matematik birebir aynidir.
        """
        # CLS token as template vector — clean global descriptor
        t_vec = F.normalize(ref.mean_cls, dim=-1)  # (1, C)

        level_maps = []
        for name, size in self.FPN_LEVELS:
            frame_patches, gh, gw = frame_feats[name]
            C = frame_patches.shape[-1]

            flat   = frame_patches[0].reshape(-1, C)           # (N_f, C)
            flat_n = F.normalize(flat, dim=-1)                 # (N_f, C)
            sims   = (flat_n @ t_vec[0].unsqueeze(-1)).squeeze(-1)  # (N_f,)
            sim_map = sims.reshape(gh, gw).cpu().numpy()       # (gh, gw)

            # upsample to original resolution
            order = self.config['fpn']['upsample_order']
            up = nd_zoom(sim_map, (H_orig / gh, W_orig / gw), order=order)
            level_maps.append(up)

        stack = np.stack(level_maps, axis=0)  # (3, H, W)
        if self.fusion == "max":
            return stack.max(axis=0)
        elif self.fusion == "mean":
            return stack.mean(axis=0)
        else:  # weighted
            w = np.array(self.config['fpn']['fusion_weights'])
            return (stack * w[:, None, None]).sum(axis=0)
        # ── Module 5: BBox Extraction ─────────────────────────────────────────────

    def _extract_bbox(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        object_id: int,
    ) -> Optional[Detection]:
        """
        Extract bounding box from heatmap using absolute thresholding
        and morphological closing. Point-in-polygon to isolate peak contour.
        """
        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        cx, cy = max_loc

        # confidence gate
        if max_val < self.min_confidence:
            return None

        # absolute threshold — prevents noise amplification
        raw_threshold = max(max_val * self.threshold_ratio, self.noise_floor)
        _, thresh_map = cv2.threshold(heatmap, raw_threshold, 1.0, cv2.THRESH_BINARY)
        thresh_8u = np.uint8(thresh_map * 255)

        # morphological closing to fill holes
        shape_str = self.config['matching']['morphology']['kernel_shape']
        shape_map = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
        shape = shape_map.get(shape_str, cv2.MORPH_ELLIPSE)
        k_size = self.config['matching']['morphology']['kernel_size']
        kernel = cv2.getStructuringElement(shape, (k_size, k_size))
        closed = cv2.morphologyEx(thresh_8u, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # find contour containing the peak
        best = None
        for c in contours:
            if cv2.pointPolygonTest(c, (cx, cy), measureDist=False) >= 0:
                best = c
                break
        if best is None:
            best = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(best)

        return Detection(
            object_id=object_id,
            x_center=x + w / 2,
            y_center=y + h / 2,
            width=w,
            height=h,
            confidence=float(max_val),
        )

    # ── Module 6: NMS ─────────────────────────────────────────────────────────

    def _nms(self, detections: List[Detection], iou_threshold: float = None) -> List[Detection]:
        if iou_threshold is None:
            iou_threshold = self.config['matching']['nms_iou_threshold']
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept = []
        for det in detections:
            if not any(self._iou(det, k) > iou_threshold for k in kept):
                kept.append(det)
        return kept

    def _iou(self, a: Detection, b: Detection) -> float:
        ax1, ay1, ax2, ay2 = a.to_xyxy()
        bx1, by1, bx2, by2 = b.to_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0 else 0.0

    # ── Main Inference ────────────────────────────────────────────────────────

    def match(
        self,
        frame: np.ndarray,
        save_path: str = None,
        visualize: bool = False,
    ) -> List[Detection]:
        """
        Run full template matching pipeline on a frame.
        Returns list of Detection objects.
        """
        if not self.references:
            raise ValueError("No references registered. Call add_reference() first.")

        H_orig, W_orig = frame.shape[:2]

        # Kare ozniteliklerini olcek basina BIR KEZ cikar (referanstan bagimsiz).
        # Boylece N referans icin 3 (olcek) omurga gecisi yeter — eskiden 3*N idi.
        frame_feats = {}
        for name, size in self.FPN_LEVELS:
            _, frame_patches, gh, gw = self._extract(frame, size)
            frame_feats[name] = (frame_patches, gh, gw)

        # Bypass the flawed global CLS filter and evaluate all references
        candidates = self.references

        all_detections = []
        all_heatmaps   = {}

        for ref in candidates:
            heatmap = self._fpn_heatmap(H_orig, W_orig, ref, frame_feats)
            all_heatmaps[ref.object_id] = heatmap
            det = self._extract_bbox(frame, heatmap, ref.object_id)
            if det is not None:
                all_detections.append(det)

        final = self._nms(all_detections)

        if save_path or visualize:
            self._save(frame, final, all_heatmaps, save_path, visualize)

        return final

    # ── Visualization & Save ──────────────────────────────────────────────────

    def _save(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        heatmaps: dict,
        save_path: str = None,
        visualize: bool = False,
    ):
        import matplotlib.pyplot as plt

        n_refs = len(heatmaps)
        fig, axes = plt.subplots(1, 2 + n_refs, figsize=(6 * (2 + n_refs), 5))

        # original frame
        axes[0].imshow(frame)
        axes[0].set_title("Frame")
        axes[0].axis("off")

        # heatmaps per reference
        for i, (obj_id, heatmap) in enumerate(heatmaps.items()):
            axes[1 + i].imshow(frame)
            im = axes[1 + i].imshow(heatmap, alpha=0.6, cmap="jet", vmin=0.0, vmax=0.6)
            plt.colorbar(im, ax=axes[1 + i], fraction=0.046, pad=0.04)
            axes[1 + i].set_title(f"Heatmap obj={obj_id}")
            axes[1 + i].axis("off")

        # detections
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.to_xyxy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cx, cy = int(det.x_center), int(det.y_center)
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
            label = f"obj{det.object_id} {det.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        axes[-1].imshow(vis)
        axes[-1].set_title("Detections")
        axes[-1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved → {save_path}")
        if visualize:
            plt.show()
        plt.close()

        for det in detections:
            print(f"  obj={det.object_id} center=({det.x_center:.0f},{det.y_center:.0f}) "
                  f"size=({det.width}x{det.height}) conf={det.confidence:.3f}")
