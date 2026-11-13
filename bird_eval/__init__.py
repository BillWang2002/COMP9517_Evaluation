"""Unified A/B evaluation, timing and demonstration; no training."""
import os
from pathlib import Path

_cache=Path(__file__).resolve().parents[1]/'.cache'
os.environ.setdefault('YOLO_CONFIG_DIR',str(_cache/'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(_cache/'matplotlib'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')

