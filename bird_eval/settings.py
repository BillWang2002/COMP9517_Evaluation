"""Fixed validation-selected settings. Never choose parameters using test."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/beach_bird'
MODELS={
    'A1':{'checkpoint':'weights/A/A1.pth','sha256':'25001996ed2629ec0ba9e7cb3c4610180a71907d64e3ff5347bdeab091f24c78','score_floor':0.05,'nms_iou':0.5},
    'B0':{'checkpoint':'weights/B/B0.pt','sha256':'77da34dfcb2bfbe18369f894425460fbdb205f314a9ee2c31fe5ac017b8aa7b8','score_floor':0.001,'nms_iou':0.7},
}
INPUT_SIZE=640
CAP=300
WARMUP=5
THREADS=4
WORKING_SCORES={'A1':0.75,'B0':0.4,'FUSION':0.4}
FUSION_PARAMETERS={'a_min_score':0.9,'novelty_iou':0.5,'a_min_original_box_area':9216,
                   'a_score_scale':0.75,'b_support_min_score':0.1,'a_supplement_nms_iou':0.5}
SPLITS={'val':{'images':291,'GT':1843,'fingerprint':'c5e0e9bfa54eb6dce5acf8718fe129378244dbf3f2050fe121933ef683e3c34c'},
        'test':{'images':240,'GT':1518,'fingerprint':'ff9beb5537c020eaf17847d42dcd1fda1ca41203b3ee31d5263f386e9ede6dc9'}}
TIMING_SCOPE=('prepared RGB PIL image to original-coordinate CPU boxes/scores/labels; includes input conversion, '
              'device transfers, model forward/transforms, internal NMS, output copies and selective fusion when enabled; '
              'excludes decoding, labels, model loading, warmup, metric calculation and rendering')

