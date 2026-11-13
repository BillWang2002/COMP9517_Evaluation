"""Checkpoint-compatible Faster R-CNN construction for evaluation only."""
from __future__ import annotations

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process.")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()




def _frozen_normalization(module):
    """Recreate checkpoint-compatible normalization without fetching COCO weights.

    This inference-only factory immediately loads a complete fine-tuned state
    dict. Training freeze policies and COCO initialization are not needed here.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=0.0))
        else:
            _frozen_normalization(child)


def make_model(legacy_eval=False):
    # Source A uses frozen normalization with eps=0 for canonical evaluation.
    # Rebuild that structure, then load the full A0/A1 checkpoint strictly.
    # No pretrained initialization is downloaded by this evaluation-only factory.
    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        min_size=640, max_size=640, box_detections_per_img=300,
    )
    if not legacy_eval:
        _frozen_normalization(model)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model
