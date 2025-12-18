import ast
import os
import glob
import gc
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import argparse
import torch
from torch.amp import autocast, GradScaler
import json
from joblib import Parallel, delayed
import ast
from time import time
from utils.utils import process_one_video_preds_multiclass
from torch.utils.data import DataLoader
from utils.utils import process_one_video_preds_vectorized, process_one_video_preds_filters
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-G' , '--gpu', type=str, default='0', help='GPU id to use')

# Add project directories to path
BASEDIR = './'
for DIRNAME in 'configs data models postprocess metrics split_data'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

start_time = time()
parser_args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = parser_args.gpu

cfg = copy(importlib.import_module("cfg_1").cfg)
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {cfg.device}")


Net233 = importlib.import_module("mdl_233").Net
Net234 = importlib.import_module("mdl_234").Net
Net238 = importlib.import_module("mdl_238").Net
Net240 = importlib.import_module("mdl_240").Net
Net242 = importlib.import_module("mdl_242").Net
Net243 = importlib.import_module("mdl_243").Net
Net244 = importlib.import_module("mdl_244").Net
Net245 = importlib.import_module("mdl_245").Net
Net256 = importlib.import_module("mdl_256").Net

CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn

window_size = cfg.window_size
cfg.stride = window_size // 2  # 50% overlap
min_duration = 5 # Minimum frames for a behavior

cfg_233 = copy(cfg)
cfg_233.reverse_time = False
cfg_233.use_gnn = False
cfg_233.use_bn = True
cfg_233.cnn_extractor = True
cfg_233.encoder_config.input_dim=256
cfg_233.encoder_config.encoder_dim=256
cfg_233.encoder_config.num_layers=4
cfg_233.encoder_config.num_attention_heads=4
cfg_233.per_mouse_feature_dim = cfg_233.feature_dim // 4

model_233 = Net233(cfg_233).to(cfg_233.device)
model_233 = torch.compile(model_233)
weights_path_233 = "output/mabe-weights/MAB-233_best.pth"
weights_233 = torch.load(weights_path_233, map_location=cfg_233.device, weights_only=False)#['model_state_dict']
model_233.load_state_dict(weights_233)
model_233.eval()

cfg_238 = copy(cfg)
cfg_238.cnn_extractor = True
cfg_238.use_gnn = True
cfg_238.use_bn = True
cfg_238.encoder_config.input_dim=256
cfg_238.encoder_config.encoder_dim=256
cfg_238.encoder_config.num_layers=4
cfg_238.encoder_config.num_attention_heads=4
cfg_238.per_mouse_feature_dim = cfg_238.feature_dim // 4

model_238 = Net238(cfg_238).to(cfg_238.device)
model_238 = torch.compile(model_238)
weights_path_238 = "output/mabe-weights/MAB-238_best.pth"
weights_238 = torch.load(weights_path_238, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_238.load_state_dict(weights_238)
model_238.eval()

model_289 = Net238(cfg_238).to(cfg_238.device)
model_289 = torch.compile(model_289)
weights_path_289 = "output/mabe-weights/MAB-289_best.pth"
weights_289 = torch.load(weights_path_289, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_289.load_state_dict(weights_289)
model_289.eval()

model_290 = Net238(cfg_238).to(cfg_238.device)
model_290 = torch.compile(model_290)
weights_path_290 = "output/mabe-weights/MAB-290_best.pth"
weights_290 = torch.load(weights_path_290, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_290.load_state_dict(weights_290)
model_290.eval()

model_293 = Net238(cfg_238).to(cfg_238.device)
model_293 = torch.compile(model_293)
weights_path_293 = "output/mabe-weights/MAB-293_best.pth"
weights_293 = torch.load(weights_path_293, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_293.load_state_dict(weights_293)
model_293.eval()

model_292 = Net238(cfg_238).to(cfg_238.device)
model_292 = torch.compile(model_292)
weights_path_292 = "output/mabe-weights/MAB-292_best.pth"
weights_292 = torch.load(weights_path_292, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_292.load_state_dict(weights_292)
model_292.eval()

model_245 = Net245(cfg_238).to(cfg_238.device)
model_245 = torch.compile(model_245)
weights_path_245 = "output/mabe-weights/MAB-245_best.pth"
weights_245 = torch.load(weights_path_245, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_245.load_state_dict(weights_245)
model_245.eval()

model_264 = Net245(cfg_238).to(cfg_238.device)
model_264 = torch.compile(model_264)
weights_path_264 = "output/mabe-weights/MAB-264_best.pth"
weights_264 = torch.load(weights_path_264, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_264.load_state_dict(weights_264)
model_264.eval()

model_266 = Net245(cfg_238).to(cfg_238.device)
model_266 = torch.compile(model_266)
weights_path_266 = "output/mabe-weights/MAB-266_best.pth"
weights_266 = torch.load(weights_path_266, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_266.load_state_dict(weights_266)
model_266.eval()

model_267 = Net245(cfg_238).to(cfg_238.device)
model_267 = torch.compile(model_267)
weights_path_267 = "output/mabe-weights/MAB-267_best.pth"
weights_267 = torch.load(weights_path_267, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_267.load_state_dict(weights_267)
model_267.eval()

model_269 = Net245(cfg_238).to(cfg_238.device)
model_269 = torch.compile(model_269)
weights_path_269 = "output/mabe-weights/MAB-269_best.pth"
weights_269 = torch.load(weights_path_269, map_location=cfg_238.device, weights_only=False)#['model_state_dict']
model_269.load_state_dict(weights_269)
model_269.eval()

cfg_244 = copy(cfg_238)
cfg_244.per_mouse_feature_dim = 304 // 4
model_244 = Net244(cfg_244).to(cfg_244.device)
model_244 = torch.compile(model_244)
weights_path_244 = "output/mabe-weights/MAB-244_best.pth"
weights_244 = torch.load(weights_path_244, map_location=cfg_244.device, weights_only=False)#['model_state_dict']
model_244.load_state_dict(weights_244)
model_244.eval()

cfg_234 = copy(cfg)
cfg_234.reverse_time = False
cfg_234.use_gnn = False
cfg_234.use_bn = True
cfg_234.cnn_extractor = True
cfg_234.encoder_config.input_dim=256
cfg_234.encoder_config.encoder_dim=256
cfg_234.encoder_config.num_layers=4
cfg_234.encoder_config.num_attention_heads=4
cfg_234.per_mouse_feature_dim = cfg_234.feature_dim // 4

model_234 = Net234(cfg_234).to(cfg_234.device)
model_234 = torch.compile(model_234)
weights_path_234 = "output/mabe-weights/MAB-234_best.pth"
weights_234 = torch.load(weights_path_234, map_location=cfg_234.device, weights_only=False)#['model_state_dict']
model_234.load_state_dict(weights_234)
model_234.eval()

cfg_240 = copy(cfg)
cfg_240.cnn_extractor = True
cfg_240.use_gnn = False
cfg_240.use_bn = True
cfg_240.encoder_config.input_dim=256
cfg_240.encoder_config.encoder_dim=256
cfg_240.encoder_config.num_layers=8
cfg_240.encoder_config.num_attention_heads=4
cfg_240.per_mouse_feature_dim = cfg_240.feature_dim // 4

model_240 = Net240(cfg_240).to(cfg_240.device)
model_240 = torch.compile(model_240)
weights_path_240 = "output/mabe-weights/MAB-240_best.pth"
weights_240 = torch.load(weights_path_240, map_location=cfg_240.device, weights_only=False)#['model_state_dict']
model_240.load_state_dict(weights_240)
model_240.eval()

cfg_242 = copy(cfg)
cfg_242.cnn_extractor = True
cfg_242.use_gnn = True
cfg_242.use_bn = True
cfg_242.encoder_config.input_dim=256
cfg_242.encoder_config.encoder_dim=256
cfg_242.encoder_config.num_layers=4
cfg_242.encoder_config.num_attention_heads=4
cfg_242.per_mouse_feature_dim = 40 #cfg_242.feature_dim // 4

model_242 = Net242(cfg_242).to(cfg_242.device)
model_242 = torch.compile(model_242)
weights_path_242 = "output/mabe-weights/MAB-242_best.pth"
weights_242 = torch.load(weights_path_242, map_location=cfg_242.device, weights_only=False)#['model_state_dict']
model_242.load_state_dict(weights_242)
model_242.eval()

cfg_243 = copy(cfg)
cfg_243.cnn_extractor = True
cfg_243.use_gnn = True
cfg_243.use_bn = True
cfg_243.encoder_config.input_dim=256
cfg_243.encoder_config.encoder_dim=256
cfg_243.encoder_config.num_layers=4
cfg_243.encoder_config.num_attention_heads=4
cfg_243.per_mouse_feature_dim = 40 #cfg.feature_dim // len(cfg.set_mice)

model_243 = Net243(cfg_243).to(cfg_243.device)
model_243 = torch.compile(model_243)
weights_path_243 = "output/mabe-weights/MAB-243_best.pth"
weights_243 = torch.load(weights_path_243, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_243.load_state_dict(weights_243)
model_243.eval()

model_294 = Net243(cfg_243).to(cfg_243.device)
model_294 = torch.compile(model_294)
weights_path_294 = "output/mabe-weights/MAB-294_best.pth"
weights_294 = torch.load(weights_path_294, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_294.load_state_dict(weights_294)
model_294.eval()

model_295 = Net243(cfg_243).to(cfg_243.device)
model_295 = torch.compile(model_295)
weights_path_295 = "output/mabe-weights/MAB-295_best.pth"
weights_295 = torch.load(weights_path_295, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_295.load_state_dict(weights_295)
model_295.eval()

model_297 = Net243(cfg_243).to(cfg_243.device)
model_297 = torch.compile(model_297)
weights_path_297 = "output/mabe-weights/MAB-297_best.pth"
weights_297 = torch.load(weights_path_297, map_location=cfg_243.device, weights_only=False)#['model_state_dict']
model_297.load_state_dict(weights_297)
model_297.eval()


cfg_256 = copy(cfg)
cfg_256.cnn_extractor = True
cfg_256.use_gnn = True
cfg_256.use_bn = True
cfg_256.encoder_config.input_dim=256
cfg_256.encoder_config.encoder_dim=256
cfg_256.encoder_config.num_layers=8
cfg_256.encoder_config.num_attention_heads=4

model_256 = Net256(cfg_256).to(cfg_256.device)
model_256 = torch.compile(model_256)
weights_path_256 = "output/mabe-weights/MAB-256_best.pth"
weights_256 = torch.load(weights_path_256, map_location=cfg_256.device, weights_only=False)#['model_state_dict']
model_256.load_state_dict(weights_256)
model_256.eval()

cfg_273 = copy(cfg)
cfg_273.cnn_extractor = True
cfg_273.use_gnn = False
cfg_273.use_bn = True
cfg_273.encoder_config.input_dim=256
cfg_273.encoder_config.encoder_dim=256
cfg_273.encoder_config.num_layers=4
cfg_273.encoder_config.num_attention_heads=4
cfg_273.per_mouse_feature_dim = cfg_273.feature_dim // 4

model_273 = Net233(cfg_273).to(cfg_273.device)
model_273 = torch.compile(model_273)
weights_path_273 = "output/mabe-weights/MAB-273_best.pth"
weights_273 = torch.load(weights_path_273, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_273.load_state_dict(weights_273)
model_273.eval() 

model_274 = Net233(cfg_273).to(cfg_273.device)
model_274 = torch.compile(model_274)
weights_path_274 = "output/mabe-weights/MAB-274_best.pth"
weights_274 = torch.load(weights_path_274, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_274.load_state_dict(weights_274)
model_274.eval()

model_275 = Net233(cfg_273).to(cfg_273.device)
model_275 = torch.compile(model_275)
weights_path_275 = "output/mabe-weights/MAB-275_best.pth"
weights_275 = torch.load(weights_path_275, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_275.load_state_dict(weights_275)
model_275.eval()

model_276 = Net233(cfg_273).to(cfg_273.device)
model_276 = torch.compile(model_276)
weights_path_276 = "output/mabe-weights/MAB-276_best.pth"
weights_276 = torch.load(weights_path_276, map_location=cfg_273.device, weights_only=False)#['model_state_dict']
model_276.load_state_dict(weights_276)
model_276.eval()


cfg_278 = copy(cfg)
cfg_278.cnn_extractor = True
cfg_278.use_gnn = False
cfg_278.use_bn = True
cfg_278.encoder_config.input_dim=256
cfg_278.encoder_config.encoder_dim=256
cfg_278.encoder_config.num_layers=4
cfg_278.encoder_config.num_attention_heads=4
cfg_278.per_mouse_feature_dim = cfg_278.feature_dim // 4

model_278 = Net234(cfg_278).to(cfg_278.device)
model_278 = torch.compile(model_278)
weights_path_278 = "output/mabe-weights/MAB-278_best.pth"
weights_278 = torch.load(weights_path_278, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_278.load_state_dict(weights_278)
model_278.eval()

model_279 = Net234(cfg_278).to(cfg_278.device)
model_279 = torch.compile(model_279)
weights_path_279 = "output/mabe-weights/MAB-279_best.pth"
weights_279 = torch.load(weights_path_279, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_279.load_state_dict(weights_279)
model_279.eval()

model_284 = Net234(cfg_278).to(cfg_278.device)
model_284 = torch.compile(model_284)
weights_path_284 = "output/mabe-weights/MAB-284_best.pth"
weights_284 = torch.load(weights_path_284, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_284.load_state_dict(weights_284)
model_284.eval()

model_285 = Net234(cfg_278).to(cfg_278.device)
model_285 = torch.compile(model_285)
weights_path_285 = "output/mabe-weights/MAB-285_best.pth"
weights_285 = torch.load(weights_path_285, map_location=cfg_278.device, weights_only=False)#['model_state_dict']
model_285.load_state_dict(weights_285)
model_285.eval()


### INSERT YOUR TEST SET CSV PATH HERE ###
val_df = pd.read_csv('datamount/splits/lab_stratified_val.csv')

num_mice = len(cfg.set_mice)
num_pairs = num_mice * num_mice
actions = cfg.set_behavior_classes
num_actions = len(actions)
action_map = cfg.id_to_action_map

mouse_map_str = cfg.mouse_id_to_string

chunk_size = 50  # Adjust based on memory limits; e.g., process 50 videos at a time
all_predictions_dfs = []

for i in range(0, len(val_df), chunk_size):
    chunk_df = val_df.iloc[i:i + chunk_size]
    temp_prediction_dfs = []
    val_dataset = CustomDataset(chunk_df, cfg, aug=None, mode="val")

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.batch_size_val if hasattr(cfg, 'batch_size_val') else cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=val_collate_fn if val_collate_fn else tr_collate_fn,
        drop_last=False
    )

    # --- PHASE 1: GPU-BOUND WORK (Inference) ---
    video_accum = {}  # video_id -> {'all_preds': np.array, 'counts': np.array, 'lab_id': str, 'num_frames': int}
    for batch in tqdm(val_dataloader, desc="Validation"):
        batch = batch_to_device(batch, cfg.device)
        with torch.no_grad():

            # output_233 = model_233(batch)
            # preds_batch_233 = output_233['predictions'].cpu().numpy()
    
            # output_234 = model_234(batch)
            # preds_batch_234 = output_234['predictions'].cpu().numpy() 
    
            # output_238 = model_238(batch)
            # preds_batch_238 = output_238['predictions'].cpu().numpy()  
    
            output_240 = model_240(batch)
            preds_batch_240 = output_240['predictions'].cpu().numpy()
        
            output_242 = model_242(batch)
            preds_batch_242 = output_242['predictions'].cpu().numpy()
                
            # output_243 = model_243(batch)
            # preds_batch_243 = output_243['predictions'].cpu().numpy()
                
            output_244 = model_244(batch)
            preds_batch_244 = output_244['predictions'].cpu().numpy() 
                
            # output_245 = model_245(batch)
            # preds_batch_245 = output_245['predictions'].cpu().numpy() 
                 
            output_256 = model_256(batch)
            preds_batch_256 = output_256['predictions'].cpu().numpy()  
            
            # ##################
            output_264 = model_264(batch)
            preds_batch_264 = output_264['predictions'].cpu().numpy()

            output_266 = model_266(batch)
            preds_batch_266 = output_266['predictions'].cpu().numpy()

            output_267 = model_267(batch)
            preds_batch_267 = output_267['predictions'].cpu().numpy()

            output_269 = model_269(batch)
            preds_batch_269 = output_269['predictions'].cpu().numpy()

            preds_batch_245fold = (preds_batch_264 + preds_batch_266 
                                + preds_batch_267 + preds_batch_269) / 4.0    
            ###############
            output_273 = model_273(batch)
            preds_batch_273 = output_273['predictions'].cpu().numpy()

            output_274 = model_274(batch)
            preds_batch_274 = output_274['predictions'].cpu().numpy()

            output_275 = model_275(batch)
            preds_batch_275 = output_275['predictions'].cpu().numpy()

            output_276 = model_276(batch)
            preds_batch_276 = output_276['predictions'].cpu().numpy()

            preds_batch_233fold = (preds_batch_273 + preds_batch_274 
                                + preds_batch_275 + preds_batch_276) / 4.0            
            ###############
            output_278 = model_278(batch)
            preds_batch_278 = output_278['predictions'].cpu().numpy()

            output_279 = model_279(batch)
            preds_batch_279 = output_279['predictions'].cpu().numpy()

            output_284 = model_284(batch)
            preds_batch_284 = output_284['predictions'].cpu().numpy()

            output_285 = model_285(batch)
            preds_batch_285 = output_285['predictions'].cpu().numpy()
            
            preds_batch_234fold = (preds_batch_278 + preds_batch_279 
                                + preds_batch_284 + preds_batch_285) / 4.0
            #################
            output_289 = model_289(batch)
            preds_batch_289 = output_289['predictions'].cpu().numpy()
            
            output_290 = model_290(batch)
            preds_batch_290 = output_290['predictions'].cpu().numpy()

            output_293 = model_293(batch)
            preds_batch_293 = output_293['predictions'].cpu().numpy()

            output_292 = model_292(batch)
            preds_batch_292 = output_292['predictions'].cpu().numpy()
            
            preds_batch_238fold = (preds_batch_289 + preds_batch_290 
                                   + preds_batch_293 + preds_batch_292) / 4.0
            ##################
            output_294 = model_294(batch)
            preds_batch_294 = output_294['predictions'].cpu().numpy()
            
            output_295 = model_295(batch)
            preds_batch_295 = output_295['predictions'].cpu().numpy()
            
            output_297 = model_297(batch)
            preds_batch_297 = output_297['predictions'].cpu().numpy()
            
            
            preds_batch_243fold = (preds_batch_294 + preds_batch_295 
                                   + preds_batch_297) / 3.0
            
            preds_batch = (preds_batch_233fold + preds_batch_234fold
                + preds_batch_238fold + preds_batch_240 + preds_batch_242 + preds_batch_243fold
                + preds_batch_244 + preds_batch_245fold + preds_batch_256) / 9.0
            


        bs = preds_batch.shape[0]
        for i in range(bs):
            video_id = batch['video_id'][i]
            if video_id not in video_accum:
                num_frames = batch['num_frames'][i].item()
                lab_id = val_dataset.video_info[video_id]['lab_id']
                video_accum[video_id] = {
                    'all_preds': np.zeros((num_frames, num_pairs, num_actions), dtype=np.float32),
                    'counts': np.zeros((num_frames, 1, 1), dtype=np.float32),
                    'lab_id': lab_id,
                    'num_frames': num_frames
                }
            
            start = batch['start_frame'][i].item()
            mask = batch['input_mask'][i].cpu().numpy()  # [seq_len]
            actual_len = int(np.sum(mask))
            preds = preds_batch[i][:actual_len]
            
            end = min(start + actual_len, video_accum[video_id]['num_frames'])
            try:
                video_accum[video_id]['all_preds'][start:end] += preds[: (end - start)]
                video_accum[video_id]['counts'][start:end] += 1
            except:
                continue


    # Average predictions
    for video_id in video_accum:
        acc = video_accum[video_id]
        counts = np.maximum(acc['counts'], 1)
        acc['all_preds'] /= counts

    # Prepare data for post-processing
    all_video_data_for_processing = [
        {
            'video_preds': video_accum[video_id]['all_preds'],
            'video_id': video_id,
            'lab_id': video_accum[video_id]['lab_id'],
            'num_frames': video_accum[video_id]['num_frames']
        }
        for video_id in video_accum
    ]

    # --- PHASE 2: Post-Processing ---
    results = Parallel(n_jobs=-1)(
        delayed(process_one_video_preds_filters)(
            cfg,
            data['video_preds'],
            data['video_id'],
            data['lab_id'],
            data['num_frames'],
            mouse_map_str,
            action_map,
            num_pairs,
            num_actions,
            min_duration,
            smooth_sigma=0.0, 
            conf_thresh=0.0,
            max_gap=0, 
            use_median_filter=False
        ) for data in tqdm(all_video_data_for_processing, desc="Phase 2: Post-Processing")
    )
    temp_prediction_dfs = [df for df in results if df is not None]
    all_predictions_dfs.extend(temp_prediction_dfs)
    del val_dataloader
    del val_dataset
    gc.collect()

# --- Ground Truth Loading ---
all_ground_truth_dfs = []
int_to_str_map = {1: 'mouse1', 2: 'mouse2', 3: 'mouse3', 4: 'mouse4'}
for _, row in val_df.iterrows():
    video_id = row['video_id']
    lab_id = row['lab_id']
    anno_path = f'datamount/train_annotation/{lab_id}/{video_id}.parquet'
    if os.path.exists(anno_path):
        anno_df = pd.read_parquet(anno_path)
        anno_df['agent_id'] = anno_df['agent_id'].map(int_to_str_map)
        anno_df['target_id'] = anno_df['target_id'].map(int_to_str_map).fillna('self')
        
        anno_df.loc[anno_df['agent_id'] == anno_df['target_id'], 'target_id'] = 'self'

        anno_df['video_id'] = video_id
        anno_df['lab_id'] = lab_id

        # Add behaviors_labeled
        behaviors_labeled = ast.literal_eval(row['behaviors_labeled']) if isinstance(row['behaviors_labeled'], str) else row['behaviors_labeled']

        anno_df['behaviors_labeled'] = json.dumps(behaviors_labeled)

        all_ground_truth_dfs.append(anno_df)

# --- PHASE 3: METRIC CALCULATION ---
FINAL_SUBMISSION_COLS = [
    'row_id',
    'video_id',
    'agent_id',
    'target_id',
    'action',
    'start_frame',
    'stop_frame'
]
if all_predictions_dfs:
    predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    predictions_df['row_id'] = range(len(predictions_df))
    predictions_df = predictions_df[FINAL_SUBMISSION_COLS]
else:
    predictions_df = pd.DataFrame(columns=FINAL_SUBMISSION_COLS)

if all_ground_truth_dfs:
    ground_truth_df = pd.concat(all_ground_truth_dfs, ignore_index=True)
    ground_truth_df['row_id'] = range(len(ground_truth_df))
else:
    ground_truth_df = pd.DataFrame()

# Calculate official F1 metric
f1_score = 0.0
if not predictions_df.empty and not ground_truth_df.empty:
    try:
        from metrics.metric_1 import mouse_fbeta
        predictions_df.to_csv("temp_predictions.csv", index=False)
        ground_truth_df.to_csv("temp_ground_truth.csv", index=False)
        f1_score = mouse_fbeta(ground_truth_df, predictions_df, beta=1)

        print(f"\n{'='*40}")
        print(f"Official F1 Score: {f1_score:.4f}")
        print(f"Total predicted intervals: {len(predictions_df)}")
        print(f"Total ground truth intervals: {len(ground_truth_df)}")

        if len(predictions_df) > 0:
            print(f"\nTop predicted actions:")
            action_counts = predictions_df['action'].value_counts().head(5)
            for action, count in action_counts.items():
                print(f"  {action}: {count}")
        print(f"{'='*40}\n")

    except Exception as e:
        print(f"Error calculating official metric: {e}")
        f1_score = 0.0
else:
    print(f"No valid predictions ({len(predictions_df)}) or ground truth ({len(ground_truth_df)})")
end_time = time()
print(f"Total inference and post-processing time: {end_time - start_time:.2f} seconds")
