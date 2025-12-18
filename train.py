import ast
import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import argparse
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import transformers
import neptune
from neptune.utils import stringify_unsupported
import json
from joblib import Parallel, delayed
from decouple import config
import ast
from utils.awp import AdvWeightPerturb
from torch.optim.swa_utils import AveragedModel, SWALR
from utils.utils import BNUpdateWrapper, custom_update_bn as update_bn


# Add project directories to path
BASEDIR = './'
for DIRNAME in 'configs data models postprocess metrics split_data'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

from utils.utils import set_seed, seed_worker, process_one_video_preds_vectorized

parser = argparse.ArgumentParser(description="MABe Training with Squeezeformer")
parser.add_argument("-C", "--config", help="config filename", default="cfg_1")
parser.add_argument("-G", "--gpu_id", default="1", help="GPU ID")
parser.add_argument("-S", "--split", default="lab_stratified",
                    help="Split name (lab_stratified, lab_holdout, gkfold_fold1)")
parser.add_argument("-F", "--fold", default=0, type=int, help="Fold number for cross-validation")

parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

if parser_args.gpu_id != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

if len(other_args) > 1:
    other_args = {k.replace('-',''): v for k, v in zip(other_args[1::2], other_args[2::2])}

    for key in other_args:
        if key in cfg.__dict__:
            print(f'Overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000)
print(f"Seed: {cfg.seed}")
set_seed(cfg.seed)

calc_metric = importlib.import_module(cfg.metric).calc_metric
Net = importlib.import_module(cfg.model).Net
CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn

split_dir = 'datamount/splits'
split_name = parser_args.split

train_df_path = os.path.join(split_dir, f'{split_name}_train.csv')
val_df_path = os.path.join(split_dir, f'{split_name}_val.csv')
fold = parser_args.fold
lab_split = "datamount/splits/lab_stratified_crossfold"
if lab_split:
    train_df_path = f"{lab_split}/train_fold{fold}.csv"
    val_df_path = f"{lab_split}/val_fold{fold}.csv"
train_df = pd.read_csv(train_df_path)
val_df = pd.read_csv(val_df_path)

print(f"Data Split: {split_name}")
print(f"Train videos: {len(train_df)} from {train_df['lab_id'].nunique()} labs")
print(f"Val videos: {len(val_df)} from {val_df['lab_id'].nunique()} labs")

if cfg.logging:
    fns = [parser_args.config] + [getattr(cfg, s) for s in 'dataset model metric'.split() if hasattr(cfg, s)]
    fns = sum([glob.glob(f"{BASEDIR}/*/{fn}.py") for fn in fns], glob.glob("./*.py"))

    if cfg.neptune_project == "common/quickstarts":
        neptune_api_token = neptune.ANONYMOUS_API_TOKEN
    else:
        neptune_api_token = config('NEPTUNE_API_TOKEN')
    try:
        neptune_run = neptune.init_run(
            project=cfg.neptune_project,
            tags=cfg.tags if hasattr(cfg, 'tags') else "mabe-squeezeformer-multiclass",
            mode=cfg.neptune_connection_mode if hasattr(cfg, 'neptune_connection_mode') else "async",
            api_token=neptune_api_token,
            capture_stdout=False,
            capture_stderr=False,
            source_files=fns,
            description=cfg.comment,
        )
        print(f"Neptune system id : {neptune_run._sys_id}")
        print(f"Neptune URL       : {neptune_run.get_url()}")
        neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)
    except Exception as e:
        print(f"Neptune initialization failed: {e}")
        neptune_run = None
else:
    neptune_run = None
    
if neptune_run is not None:
    os.makedirs(f"{cfg.output_dir}/{neptune_run._sys_id}", exist_ok=True)
    print("Created logging directory:", f"{cfg.output_dir}/{neptune_run._sys_id}")
    cfg.output_dir = f"{cfg.output_dir}/{neptune_run._sys_id}"

cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {cfg.device}")

train_dataset = CustomDataset(train_df, cfg, aug=None, mode="train")
val_dataset = CustomDataset(val_df, cfg, aug=None, mode="val")

print(f"Train dataset size: {len(train_dataset)} samples")
print(f"Val dataset size: {len(val_dataset)} samples")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=tr_collate_fn,
    drop_last=False,
    worker_init_fn=seed_worker
)

val_dataloader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=cfg.batch_size_val if hasattr(cfg, 'batch_size_val') else cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=val_collate_fn if val_collate_fn else tr_collate_fn,
    drop_last=False
)

model = Net(cfg).to(cfg.device)
model = torch.compile(model, disable=not cfg.compile)
if cfg.swa:
    model_swa = AveragedModel(model)
    model_swa = torch.compile(model_swa, disable=not cfg.compile)


total_steps = len(train_dataloader) * cfg.epochs
warmup_steps = int(cfg.warmup * len(train_dataloader)) if cfg.warmup < 1 else cfg.warmup
optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)

if cfg.scheduler == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=cfg.lr,
                                                        pct_start=0.15,
                                                        steps_per_epoch=len(train_dataloader), 
                                                        epochs=cfg.epochs)
elif cfg.scheduler == "cosine":
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
if cfg.swa:
    swa_scheduler = None #SWALR(optimizer, swa_lr=cfg.swa_lr)
else:
    swa_scheduler = None

scaler = GradScaler(device=cfg.device) if cfg.mixed_precision else None
output_dir = f"{cfg.output_dir}/fold{parser_args.fold}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if hasattr(cfg, 'awp') and cfg.awp:
    awp = AdvWeightPerturb(
        model=model,
        delta=getattr(cfg, 'awp_delta', 0.1),
        eps=getattr(cfg, 'awp_eps', 1e-6),
        use_mixed_precision=cfg.mixed_precision,
        scaler=scaler,
        cfg=cfg
    )
    # Optional: Define start epoch for AWP
    awp_start_epoch = getattr(cfg, 'awp_start_epoch', 0)
    print(f"AWP enabled with delta={awp.delta}, starting from epoch {awp_start_epoch}")
else:
    awp = None

# Training variables
best_val_loss = float('inf')
best_val_metric = 0
cfg.curr_step = 0
i = 0

for epoch in range(cfg.epochs):
    cfg.curr_epoch = epoch
    model.train()
    train_losses = []
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{cfg.epochs} [Train]')

    for batch_idx, batch in enumerate(progress_bar):
        i += 1
        cfg.curr_step += cfg.batch_size
        batch = batch_to_device(batch, cfg.device)

        if awp is not None and epoch >= awp_start_epoch:
            if cfg.grad_accumulation > 1:
                raise ValueError("AWP is not implemented with gradient accumulation > 1. Set cfg.grad_accumulation=1.")
            
            loss = awp.train_step(optimizer, batch)
            if scheduler is not None:
                scheduler.step()
        else:
            # Normal training
            if cfg.mixed_precision and scaler is not None:
                with autocast(device_type=cfg.device):
                    output_dict = model(batch)
                    loss = output_dict["loss"]
            else:
                output_dict = model(batch)
                loss = output_dict["loss"]

            if cfg.grad_accumulation > 1:
                loss = loss / cfg.grad_accumulation

            # Backward pass
            if cfg.mixed_precision and scaler is not None:
                scaler.scale(loss).backward()

                if i % cfg.grad_accumulation == 0:
                    if cfg.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                loss.backward()

                if i % cfg.grad_accumulation == 0:
                    if cfg.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()


            if epoch >= cfg.swa_start and swa_scheduler:
                swa_scheduler.step()
            elif scheduler:
                scheduler.step()

        train_losses.append(loss.item())
        
        progress_bar.set_postfix({
            'loss': np.mean(train_losses[-100:]) if train_losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

        if neptune_run and batch_idx % 10 == 0:
            neptune_run["train/loss"].append(loss.item())
            neptune_run["train/lr"].append(optimizer.param_groups[0]['lr'])

    if epoch >= cfg.swa_start and cfg.swa:
        model_swa.update_parameters(model)  # Now outside batch loop: once per epoch

    avg_train_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

    if (epoch + 1) % cfg.eval_epochs == 0:
        eval_model = model
        if cfg.swa_eval and epoch >= cfg.swa_start and cfg.swa:
            update_bn(BNUpdateWrapper(train_dataloader), model_swa, device=cfg.device)  # Update BN stats for SWA
            eval_model = model_swa
            print(f"Using SWA model for evaluation (epoch {epoch+1})")
        
        eval_model.eval()

        window_size = cfg.window_size
        min_duration = 1 

        num_mice = len(cfg.set_mice)
        num_pairs = num_mice * num_mice
        actions = cfg.set_behavior_classes
        num_actions = len(actions)
        action_map = cfg.id_to_action_map

        mouse_map_str = cfg.mouse_id_to_string

        # --- PHASE 1: GPU-BOUND WORK (Inference) ---
        video_accum = {}  # video_id -> {'all_preds': np.array, 'counts': np.array, 'lab_id': str, 'num_frames': int}
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = batch_to_device(batch, cfg.device)
            with torch.no_grad():
                output = eval_model(batch)
                preds_batch = output['predictions'].cpu().numpy()  # [bs, seq_len, num_pairs, num_actions]
            
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
                if start < 0:
                    start = 0
                    
                mask = batch['input_mask'][i].cpu().numpy()  # [seq_len]
                actual_len = int(np.sum(mask))
                preds = preds_batch[i][:actual_len]
                if cfg.reverse_time:
                    preds = np.flip(preds, axis=0)
                
                end = min(start + actual_len, video_accum[video_id]['num_frames'])
                video_accum[video_id]['all_preds'][start:end] += preds[: (end - start)]
                video_accum[video_id]['counts'][start:end] += 1
        
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
            delayed(process_one_video_preds_vectorized)(
                cfg,
                data['video_preds'],
                data['video_id'],
                data['lab_id'],
                data['num_frames'],
                mouse_map_str,
                action_map,
                num_pairs,
                num_actions,
                min_duration
            ) for data in tqdm(all_video_data_for_processing, desc="Phase 2: Post-Processing")
        )
        all_predictions_dfs = [df for df in results if df is not None]

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
        # (Remains unchanged from your original code)
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

                predictions_df.to_csv(f"{output_dir}/temp_val_predictions.csv", index=False)
                ground_truth_df.to_csv(f"{output_dir}/temp_ground_truth.csv", index=False)
                f1_score = mouse_fbeta(ground_truth_df, predictions_df, beta=1)

                print(f"\n{'='*40}")
                print(f"Epoch {epoch+1} - Official F1 Score: {f1_score:.4f}")
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

        # Log to Neptune
        if neptune_run:
            neptune_run["val/f1_score"].append(f1_score)
            # neptune_run["val/num_predictions"].append(len(predictions_df))

        if not predictions_df.empty:
            predictions_df.to_csv(f"{output_dir}/latest_val_predictions.csv", index=False)
            print(f"Saved *latest* validation predictions to {output_dir}/latest_val_predictions.csv")

        # Save best model based on F1 score
        if f1_score > best_val_metric:
            best_val_metric = f1_score
            # Save the appropriate model based on what was used for eval
            save_model = eval_model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': f1_score,
                'cfg': cfg
            }, f"{output_dir}/best_f1_model.pth")
            print(f"Saved best F1 model (F1: {best_val_metric:.4f})")

            # Save predictions for analysis
            if not predictions_df.empty:
                predictions_df.to_csv(f"{output_dir}/best_val_predictions.csv", index=False)
                print(f"Copied best validation predictions to {output_dir}/best_val_predictions.csv")

    # Save checkpoint
    if (epoch + 1) % cfg.eval_epochs == 0 and epoch > cfg.epochs // 3:
        torch.save({'model_state_dict': model.state_dict()}, f"{output_dir}/checkpoint_epoch_{epoch+1}_model_weights.pth")
        if epoch >= cfg.swa_start and cfg.swa:
            update_bn(BNUpdateWrapper(train_dataloader), model_swa, device=cfg.device)  # Ensure BN is updated before saving SWA
            torch.save({'model_state_dict': model_swa.state_dict()}, f"{output_dir}/checkpoint_epoch_{epoch+1}_model_swa_weights.pth")
        print(f"Saved checkpoint at epoch {epoch+1}")
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if cfg.swa:
    update_bn(BNUpdateWrapper(train_dataloader), model_swa, device=cfg.device)
# Final save
torch.save({'epoch': cfg.epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'cfg': cfg}, f"{output_dir}/final_model.pth")
if cfg.swa:
    torch.save({'epoch': cfg.epochs, 'model_state_dict': model_swa.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'cfg': cfg}, f"{output_dir}/final_swa_model.pth")
    
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best validation loss: {best_val_loss:.4f}")
if best_val_metric > 0:
    print(f"Best validation metric: {best_val_metric:.4f}")
print(f"Models saved in: {output_dir}")

# Stop Neptune
if neptune_run:
    neptune_run.stop()

print("\nTraining finished successfully!")