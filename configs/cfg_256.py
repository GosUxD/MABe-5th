import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.debug = True
cfg.logging = False
cfg.comment = ""

#paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"datamount/weights/{os.path.basename(__file__).split('.')[0]}"
cfg.data_folder = f"datamount/"
cfg.train_df = f'datamount/train.csv'

# stages
cfg.test = False
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.seed = 1994

#logging
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

cfg.MASTER_SKELETON = [
    'nose',
    'ear_left',
    'ear_right',
    'head_center',
    'body_center',
    'tail_base'
]
cfg.NUM_MASTER_KEYPOINTS = len(cfg.MASTER_SKELETON)
cfg.MASTER_SKELETON_MAP = {name: i for i, name in enumerate(cfg.MASTER_SKELETON)}

cfg.tracked_bodyparts = ['body_center', 'ear_left', 'ear_right', 'forepaw_left', 'forepaw_right', 
    'head', 'headpiece_bottombackleft', 'headpiece_bottombackright', 
    'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft',
    'headpiece_topfrontright', 'hindpaw_left', 'hindpaw_right', 'hip_left',
    'hip_right', 'lateral_left', 'lateral_right', 'neck', 'nose', 'spine_1', 
    'spine_2', 'tail_base', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint', 'tail_tip'
]

cfg.set_behavior_classes = ['allogroom', 'approach', 'attack', 'attemptmount', 'avoid', 'biteobject',
    'chase', 'chaseattack', 'climb', 'defend', 'dig', 'disengage', 'dominance', 'dominancegroom', 
    'dominancemount', 'ejaculate', 'escape', 'exploreobject', 'flinch', 'follow', 'freeze', 'genitalgroom',
    'huddle', 'intromit', 'mount', 'rear', 'reciprocalsniff', 'rest', 'run', 'self', 'selfgroom', 
    'shepherd', 'sniff', 'sniffbody', 'sniffface', 'sniffgenital', 'submit', 'tussle', 'no_action'
]

cfg.set_mice = ['mouse1', 'mouse2', 'mouse3', 'mouse4']
cfg.max_pairs = len(cfg.set_mice) * len(cfg.set_mice)  # 16 if 4 mice

cfg.mouse_id_map = {1:0, 2:1, 3:2, 4:3}
cfg.mouse_id_to_string = {0:'mouse1', 1:'mouse2', 2:'mouse3', 3:'mouse4'}

cfg.action_id_map = {'allogroom': 0, 'approach': 1, 'attack': 2, 'attemptmount': 3, 'avoid': 4,
                     'biteobject': 5, 'chase': 6, 'chaseattack': 7, 'climb': 8, 'defend': 9, 'dig': 10,
                     'disengage': 11, 'dominance': 12, 'dominancegroom': 13, 'dominancemount': 14,
                     'ejaculate': 15, 'escape': 16, 'exploreobject': 17, 'flinch': 18, 'follow': 19,
                     'freeze': 20, 'genitalgroom': 21, 'huddle': 22, 'intromit': 23, 'mount': 24,
                     'rear': 25, 'reciprocalsniff': 26, 'rest': 27, 'run': 28, 'self': 29,
                     'selfgroom': 30, 'shepherd': 31, 'sniff': 32, 'sniffbody': 33, 'sniffface': 34,
                     'sniffgenital': 35, 'submit': 36, 'tussle': 37, 'no_action': 38}

cfg.id_to_action_map = {v:k for k,v in cfg.action_id_map.items()}

# DETECTED FROM ACTUAL DATASET - actions that only occur when agent_id == target_id
cfg.self_only_actions = ['biteobject', 'climb', 'dig', 'exploreobject', 'freeze', 'genitalgroom', 'huddle', 'rear', 'rest', 'run', 'selfgroom']

# DETECTED FROM ACTUAL DATASET - actions that only occur when agent_id != target_id
cfg.social_only_actions = ['allogroom', 'approach', 'attack', 'attemptmount', 'avoid', 'chase', 'chaseattack', 
                           'defend', 'disengage', 'dominance', 'dominancegroom', 'dominancemount', 'ejaculate', 
                           'escape', 'flinch', 'follow', 'intromit', 'mount', 'reciprocalsniff', 'shepherd',
                           'sniff', 'sniffbody', 'sniffface', 'sniffgenital', 'submit', 'tussle']



# DATASET
cfg.dataset = "ds_1"
    
cfg.min_windows_per_video = 5 
cfg.max_windows_per_video = 1000
cfg.windows_per_epoch_ratio = 1.0 
cfg.bias_prob = 0.5 
cfg.balanced_lab_sampling = False 
cfg.windows_per_lab_per_epoch = 500 

cfg.window_size = 512
cfg.stride = cfg.window_size // 2
cfg.feature_dim = 232
cfg.per_mouse_feature_dim = cfg.feature_dim // len(cfg.set_mice)
cfg.oversample_factor = 10
cfg.preprocessing_basedir = "datamount/preprocessed_master_skeleton_2"
# cfg.preprocessing_basedir = "datamount/preprocessed"

#model
cfg.model = "mdl_1_inter"

encoder_config = SimpleNamespace(**{})
encoder_config.input_dim=256
encoder_config.encoder_dim=256
encoder_config.num_layers=8
encoder_config.num_attention_heads= 4
encoder_config.feed_forward_expansion_factor=2
encoder_config.conv_expansion_factor= 2
encoder_config.input_dropout_p= 0.35
encoder_config.feed_forward_dropout_p= 0.35
encoder_config.attention_dropout_p= 0.35
encoder_config.conv_dropout_p= 0.35
encoder_config.conv_kernel_size= 51

cfg.encoder_config = encoder_config

cfg.cnn_extractor = True
cfg.use_bn= True
cfg.use_gnn = False 
cfg.reverse_time = False



# LOSS SETTINGS (for handling sparse labels)
cfg.use_focal_loss = True  # Start with weighted BCE, simpler to debug
cfg.focal_alpha = 0.25  # Weight for positive/negative samples in focal loss
cfg.focal_gamma = 2.0  # Focusing parameter for focal loss
cfg.pos_weight = 10.0

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 150
cfg.eval_epochs = 5
cfg.lr = 3e-4 
cfg.optimizer = "AdamW"
cfg.scheduler = "onecycle"
cfg.weight_decay = 0.05
cfg.clip_grad = 0.
cfg.warmup = 5
cfg.batch_size = 32
cfg.batch_size_val = 32
cfg.mixed_precision = True # True
cfg.pin_memory = False
cfg.grad_accumulation = 1.
cfg.num_workers = 8

cfg.compile = True  # Set to True to enable torch.compile()
cfg.awp = False  # Set to True to enable AWP
cfg.awp_delta = 0.02  # AWP delta parameter
cfg.awp_eps = 1e-6  # AWP eps parameter
cfg.awp_start_epoch = 15  # Epoch to start AWP (optional)

cfg.swa = False  # Enable Stochastic Weight Averaging
cfg.swa_eval = False  # Use SWA model for evaluation
cfg.swa_start = 45  # Epoch to start SWA
cfg.swa_lr = 1e-4  # Learning rate for SWA

#EVAL
cfg.calc_metric = True
cfg.metric = "metric_1"
cfg.save_val_data = True

# AUGS
cfg.augment = True
cfg.aug_prob = 0.7  # Overall probability to apply any augmentation per sample
cfg.aug_noise_std = 0.02  # Std dev for Gaussian noise on keypoints (in normalized units, e.g., cm)
cfg.aug_rotation_max_deg = 30  # Max rotation angle in degrees
cfg.aug_scale_min = 0.8  # Min scale factor
cfg.aug_scale_max = 1.2  # Max scale factor
cfg.aug_flip_prob = 0.5  # Probability of horizontal flip
cfg.aug_translate_max = 0.075  # Max translation shift as fraction of coordinate range
cfg.aug_dropout_prob = 0.3  # Probability to dropout (mask) a keypoint
cfg.aug_time_warp_factor = 0.0  # Max time warping factor (stretch/compress segments)
cfg.aug_mixup_alpha = 0.0 # Alpha for mixup (if enabled)
cfg.aug_mixup_prob = 0.0 # Probability to apply mixup (requires batch-level)