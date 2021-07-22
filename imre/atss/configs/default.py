import os

from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = CN()
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.NUM_EPOCHS = 30


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.MODEL_NAME = "resnet50"
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256


# ---------------------------------------------------------------------------- #
# ATSS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ATSS = CN()
_C.MODEL.ATSS.NUM_CLASSES = 6  # the number of classes including background

# Anchor parameter
_C.MODEL.ATSS.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
_C.MODEL.ATSS.ASPECT_RATIOS = (1.0,)
_C.MODEL.ATSS.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.ATSS.STRADDLE_THRESH = 0
_C.MODEL.ATSS.OCTAVE = 2.0
_C.MODEL.ATSS.SCALES_PER_OCTAVE = 1

# Head parameter
_C.MODEL.ATSS.NUM_CONVS = 4
_C.MODEL.ATSS.USE_DCN_IN_TOWER = False

# Focal loss parameter
_C.MODEL.ATSS.LOSS_ALPHA = 0.25
_C.MODEL.ATSS.LOSS_GAMMA = 2.0

# IoU parameter to select positves
_C.MODEL.ATSS.FG_IOU_THRESHOLD = 0.5
_C.MODEL.ATSS.BG_IOU_THRESHOLD = 0.4

# topk for selecting candidate positive samples from each level
_C.MODEL.ATSS.TOPK = 9

# Weight for bbox_regression loss
_C.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

# Inference parameter
_C.MODEL.ATSS.PRIOR_PROB = 0.01
_C.MODEL.ATSS.INFERENCE_TH = 0.05
_C.MODEL.ATSS.NMS_TH = 0.6
_C.MODEL.ATSS.PRE_NMS_TOP_N = 1000


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.NUM_GPUS = 0
_C.SOLVER.IMS_PER_BATCH = 1


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 1
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100


# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False
_C.TEST.BBOX_AUG.VOTE = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
