"""Grad-CAM, and turning a heat map into a defensible sentence.

Grad-CAM shows where the model attended. It is not evidence of manipulation,
and the interface says so. What it can honestly support is a sentence like
"attention concentrated on the mouth and jaw", provided that claim is derived
from a measured quantity rather than asserted.

Two things this module gets right that a naive implementation does not:

  * Region naming uses MTCNN's five facial landmarks, mapped into crop
    coordinates by preprocess.crop_face(). A fixed coordinate mapping --
    "the bottom third of the image is the mouth" -- breaks as soon as the face
    is off-centre, rotated, or differently scaled, which is most real footage.

  * The reported quantity is the share of total CAM mass falling near each
    landmark anchor, not the single peak pixel. On a ViT-Base the token grid
    is 14x14, so one "pixel" of the CAM covers a 16x16 input patch; a single
    argmax over that is a coarse thing to build a claim on. When no region
    clearly dominates, region_report() returns None and the evidence panel
    omits the sentence rather than inventing one.

WORKS ON A TRANSFORMER, NOT JUST A CNN
--------------------------------------
Grad-CAM is written for convolutional activations shaped (B, C, H, W): it
averages the gradient over the two spatial axes to weight each channel. A ViT
block emits (B, N+1, D) -- a sequence of patch tokens with a class token in
front -- which has no spatial axes to average over, so the same arithmetic
silently produces a meaningless map instead of failing.

`vit_reshape` is the bridge. It drops the class token and folds the remaining
196 patch tokens back onto the 14x14 grid they came from, giving
(B, 768, 14, 14) -- the CNN layout the rest of the algorithm already expects.
Pass it as `reshape_transform`; without it, a ViT target layer is a bug.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Landmark order returned by MTCNN.
LM_LEFT_EYE, LM_RIGHT_EYE, LM_NOSE, LM_MOUTH_L, LM_MOUTH_R = range(5)

REGION_PHRASES = {
    "eyes": "the eye region, where blending seams and inconsistent gaze often appear",
    "nose and cheeks": "the nose and cheeks, where face-swap blending boundaries often fall",
    "mouth and jaw": "the mouth and jaw area, where lip-sync manipulation usually leaves traces",
}


class GradCAM:
    """Minimal Grad-CAM. No external dependency."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 reshape_transform=None):
        """reshape_transform maps a non-CNN activation into (B, C, H, W).

        Leave it None for a convnet. For a ViT pass `vit_reshape`; see the
        module docstring for why the default arithmetic cannot work without it.
        """
        self.model = model
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = 1,
                 out_size: Optional[int] = None) -> np.ndarray:
        """x: (1,3,H,W). Returns a HxW map normalised to [0,1].

        Runs with gradients enabled -- this cannot live inside a no_grad block,
        which is why scoring and explanation are separate passes.
        """
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = self.model(x)
            # A torchvision model returns the logits tensor; a Hugging Face one
            # returns a dataclass carrying it. Accept either.
            logits = getattr(out, "logits", out)
            logits[:, class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((out_size or x.shape[-1],) * 2, dtype=np.float32)

        activations, gradients = self.activations, self.gradients
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        size = out_size or x.shape[-1]
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        if was_training:
            self.model.train()

        lo, hi = float(cam.min()), float(cam.max())
        return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def vit_reshape(t: torch.Tensor) -> torch.Tensor:
    """(B, N+1, D) transformer tokens -> (B, D, H, W) feature map.

    Token 0 is the class token. It aggregates the whole image and belongs to no
    patch, so including it would smear a global summary across the grid. The
    remaining N tokens are the patches in row-major order, which is why a plain
    reshape to (H, W) restores their original positions: for a 224px input at
    patch size 16, N = 196 = 14 x 14.

    Returned unchanged if it is already 4-D, so the same transform can be
    passed unconditionally.
    """
    if t.dim() == 4:
        return t
    b, n, d = t.shape
    side = int(round((n - 1) ** 0.5))
    if side * side != n - 1:
        raise ValueError(
            f"{n} tokens is not one class token plus a square patch grid; "
            "this layer's output cannot be laid back onto an image.")
    return t[:, 1:, :].reshape(b, side, side, d).permute(0, 3, 1, 2)


def vit_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """The layer to hook on a ViTForImageClassification.

    layernorm_before of the final block, not the block's output. The last
    block's output feeds straight into the classifier through the class token,
    and its patch tokens have already been pooled away from the decision; the
    normalised input to that block still carries per-patch structure while
    sitting late enough to be semantic. This is the layer the reference
    Grad-CAM-for-transformers implementations target.

    RESOLVED BY SEARCH, NOT BY A FIXED PATH. `model.vit.encoder.layer[-1]` is
    the layout transformers 4.44 uses and the deployed container pins, but that
    path is library internals rather than public API and it has moved between
    versions: on a newer transformers this raised
    `'ViTModel' object has no attribute 'encoder'` for every clip in a 120-clip
    run, and because _explain() catches broadly the whole sweep degraded to
    "no map" instead of failing loudly on the first clip.

    The documented path is still tried first, so nothing changes for the pinned
    version actually serving traffic. The fallback walks named_modules() for the
    last module named `layernorm_before`, which does not care how the tree above
    it is arranged. If neither finds anything the error names what was searched,
    because a silent wrong layer would produce a plausible-looking attention map
    explaining the wrong thing -- worse than no map at all.
    """
    try:
        return model.vit.encoder.layer[-1].layernorm_before
    except AttributeError:
        pass

    candidates = [mod for name, mod in model.named_modules()
                  if name.rsplit(".", 1)[-1] == "layernorm_before"]
    if candidates:
        return candidates[-1]

    raise AttributeError(
        "Cannot locate the ViT block to hook for Grad-CAM. Tried "
        "model.vit.encoder.layer[-1].layernorm_before and a search for any "
        "module named 'layernorm_before'; neither exists on a "
        f"{type(model).__name__}. This usually means the installed transformers "
        "version restructured ViT internals -- the deployed container pins "
        "transformers==4.44.2, so match that.")


def region_report(cam: np.ndarray, landmarks: np.ndarray,
                  enrichment: float = 1.5) -> Optional[Dict]:
    """Which facial region the attention map concentrates on, if any.

    The test is ENRICHMENT, not raw share: each region's fraction of total CAM
    mass divided by the fraction of the image its disc covers. That ratio has a
    principled null -- attention spread uniformly scores exactly 1.0 for every
    region, whatever the discs' sizes -- so the cut means something fixed rather
    than being tuned to one backbone's feature-map resolution.

    Raw share cannot do this. The three discs together cover only about 40% of
    a 224px crop, so a 40%-of-total-mass cut is nearly unreachable by
    construction: it silently becomes "never name a region" rather than "name
    one when the evidence is there". That was the previous rule.

    Returns region=None when nothing clears `enrichment`, so the interface can
    stay silent instead of naming whichever region happened to come first.
    `shares` and `enrichment` are returned either way, because "attention was
    spread out" is itself a measurement worth reporting.
    """
    if cam is None or landmarks is None or len(landmarks) < 5:
        return None
    total = float(cam.sum())
    if total <= 0:
        return None

    lm = np.asarray(landmarks, dtype=np.float32)
    eyes = (lm[LM_LEFT_EYE] + lm[LM_RIGHT_EYE]) / 2.0
    mouth = (lm[LM_MOUTH_L] + lm[LM_MOUTH_R]) / 2.0
    nose = lm[LM_NOSE]
    # Jaw sits below the mouth by roughly half the nose-to-mouth distance.
    jaw = mouth + (mouth - nose) * 0.5

    iod = float(np.linalg.norm(lm[LM_RIGHT_EYE] - lm[LM_LEFT_EYE]))
    if not np.isfinite(iod) or iod <= 1:
        return None
    radius = 0.9 * iod

    h, w = cam.shape
    yy, xx = np.mgrid[0:h, 0:w]

    npix = float(h * w)

    def mass_near(points):
        """-> (share of CAM mass, share of image area) for the union of discs.

        Both are returned from one mask so the ratio between them cannot drift
        apart -- the area is measured on the same discs the mass was summed
        over, not computed from the radius analytically.
        """
        near = np.zeros((h, w), dtype=bool)
        for px, py in points:
            near |= ((xx - px) ** 2 + (yy - py) ** 2) <= radius ** 2
        return float(cam[near].sum()) / total, float(near.sum()) / npix

    anchors = {
        "eyes": [eyes],
        "nose and cheeks": [nose],
        "mouth and jaw": [mouth, jaw],
    }
    shares, areas = {}, {}
    for name, pts in anchors.items():
        shares[name], areas[name] = mass_near(pts)

    # Guard the division: a disc entirely outside the crop has zero area and
    # would otherwise produce inf and win every comparison.
    ratios = {k: (shares[k] / areas[k] if areas[k] > 0 else 0.0) for k in shares}

    base = {"shares": shares, "areas": areas, "enrichment": ratios,
            "interocular_px": iod}

    best = max(ratios, key=ratios.get)
    if ratios[best] < enrichment:
        return {"region": None, "peak_xy": None,
                "reason": f"no region reached {enrichment:.2f}x the attention "
                          "it would get from an evenly spread map",
                **base}

    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {
        "region": best,
        "phrase": REGION_PHRASES[best],
        "share": shares[best],
        "peak_xy": (int(peak[1]), int(peak[0])),
        **base,
    }


def overlay(crop_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Heat map over the crop, for the evidence panel."""
    import matplotlib
    matplotlib.use("Agg")
    # matplotlib.cm.get_cmap was removed in 3.9; this form works from 3.5 on.
    from matplotlib import colormaps

    base = np.asarray(crop_rgb, dtype=np.float32) / 255.0
    heat = colormaps["inferno"](np.clip(cam, 0, 1))[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1).astype(np.float32)
