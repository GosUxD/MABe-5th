import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from configs.cfg_1 import cfg  # Assuming cfg_1.py is in a 'configs' folder
from multiprocessing import Pool, Manager
import sys
import glob
import ast

DATA_DIR = 'datamount'
PREPROC_DIR = 'datamount/preprocessed_master_skeleton_2'
os.makedirs(PREPROC_DIR, exist_ok=True)
os.makedirs(os.path.join(PREPROC_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(PREPROC_DIR, 'test'), exist_ok=True)

# Master Skeleton (lowercase to match lab data)
MASTER_SKELETON = [
    'nose',
    'ear_left',
    'ear_right',
    'head_center',
    'body_center',
    'tail_base'
]
NUM_MASTER_KEYPOINTS = len(MASTER_SKELETON)
MASTER_SKELETON_MAP = {name: i for i, name in enumerate(MASTER_SKELETON)}

# From cfg
fixed_actions = cfg.set_behavior_classes
num_actions = len(fixed_actions)
action_map = cfg.action_id_map
NO_ACTION_IDX = action_map['no_action']
max_mice = len(cfg.set_mice)
max_pairs = cfg.max_pairs
mouse_id_map = cfg.mouse_id_map

def process_tracking_video(row, split, frame_counts_shared):
    lab_id = row['lab_id']
    video_id = row['video_id']
    pix_per_cm = row.get('pix_per_cm_approx', 1.0)
    
    if pd.isna(pix_per_cm) or pix_per_cm <= 0:
        print(f"Warning {video_id}: Invalid pix_per_cm ({pix_per_cm}), defaulting to 1.0")
        pix_per_cm = 1.0
    
    # Check for no mice from metadata
    num_mice = sum(1 for i in range(1, max_mice + 1) if pd.notna(row.get(f'mouse{i}_id')))
    if num_mice == 0:
        print(f"Skipping {video_id}: no mice")
        return
    
    # Load tracking parquet
    path = os.path.join(DATA_DIR, f'{split}_tracking', lab_id, f'{video_id}.parquet')
    if not os.path.exists(path):
        print(f"Warning: No tracking file for {video_id}")
        return
    df = pd.read_parquet(path)
    if df.empty:
        print(f"Warning: Empty tracking for {video_id}")
        return
    
    # Handle types
    df['video_frame'] = df['video_frame'].astype(int)
    df['mouse_id'] = df['mouse_id'].astype(int)
    max_frame = df['video_frame'].max()
    
    # Pivot to wide
    df_x = df.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    df_y = df.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    
    # Fill missing frames
    full_index = pd.Index(range(0, max_frame + 1), name='video_frame')
    df_x = df_x.reindex(full_index, fill_value=np.nan)
    df_y = df_y.reindex(full_index, fill_value=np.nan)
    num_frames = len(full_index)
    
    # Map mouse_ids to 0-based using cfg map
    current_mouse_levels = df_x.columns.levels[0]
    new_mouse_levels = [mouse_id_map.get(level, -1) for level in current_mouse_levels]
    if any(l < 0 for l in new_mouse_levels):
        print(f"Warning: Unknown mouse_ids in {video_id}")
    df_x.columns = df_x.columns.set_levels(new_mouse_levels, level=0)
    df_y.columns = df_y.columns.set_levels(new_mouse_levels, level=0)
    
    # Helper to get bodypart array [frames, mice], reindexed to all mice with NaN fill
    def get_bp_array(bp, is_x=True, fill_nan=True):
        df = df_x if is_x else df_y
        if bp not in df.columns.levels[1]:
            return np.full((num_frames, max_mice), np.nan, dtype=np.float32)
        bp_df = df.xs(bp, level=1, axis=1)
        bp_df = bp_df.reindex(columns=range(max_mice), fill_value=np.nan)
        return bp_df.values.astype(np.float32)
    
    # For master skeleton, create keypoints array
    keypoints = np.full((num_frames, max_mice, NUM_MASTER_KEYPOINTS, 2), np.nan, dtype=np.float32)
    
    # Vectorized mapping
    # Nose
    nose_x = get_bp_array('nose', is_x=True)
    nose_y = get_bp_array('nose', is_x=False)
    keypoints[:, :, MASTER_SKELETON_MAP['nose'], 0] = nose_x
    keypoints[:, :, MASTER_SKELETON_MAP['nose'], 1] = nose_y
    # Special fallback for labs like GroovyShrew
    if lab_id == 'GroovyShrew':
        head_x = get_bp_array('head', is_x=True)
        head_y = get_bp_array('head', is_x=False)
        keypoints[:, :, MASTER_SKELETON_MAP['nose'], 0] = np.where(np.isnan(nose_x), head_x, nose_x)
        keypoints[:, :, MASTER_SKELETON_MAP['nose'], 1] = np.where(np.isnan(nose_y), head_y, nose_y)
    
    # Ear left/right
    keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 0] = get_bp_array('ear_left', is_x=True)
    keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 1] = get_bp_array('ear_left', is_x=False)
    keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 0] = get_bp_array('ear_right', is_x=True)
    keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 1] = get_bp_array('ear_right', is_x=False)
    
    # Tail base
    keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 0] = get_bp_array('tail_base', is_x=True)
    keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 1] = get_bp_array('tail_base', is_x=False)
    
    # Head_Center: first head, then neck, then average ear_left/right
    head_x = get_bp_array('head', is_x=True)
    head_y = get_bp_array('head', is_x=False)
    neck_x = get_bp_array('neck', is_x=True)
    neck_y = get_bp_array('neck', is_x=False)
    ear_left_x = keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 0]
    ear_left_y = keypoints[:, :, MASTER_SKELETON_MAP['ear_left'], 1]
    ear_right_x = keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 0]
    ear_right_y = keypoints[:, :, MASTER_SKELETON_MAP['ear_right'], 1]
    
    avg_ear_x = np.nanmean(np.stack([ear_left_x, ear_right_x], axis=-1), axis=-1)
    avg_ear_y = np.nanmean(np.stack([ear_left_y, ear_right_y], axis=-1), axis=-1)
    
    head_center_x = np.where(~np.isnan(head_x), head_x,
                    np.where(~np.isnan(neck_x), neck_x, avg_ear_x))
    head_center_y = np.where(~np.isnan(head_y), head_y,
                    np.where(~np.isnan(neck_y), neck_y, avg_ear_y))
    keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 0] = head_center_x
    keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 1] = head_center_y
    
    # Body_Center: body_center, then average [spine_1, spine_2, hip_left, hip_right, neck], then (head_center + tail_base)/2
    body_center_x = get_bp_array('body_center', is_x=True)
    body_center_y = get_bp_array('body_center', is_x=False)
    
    spine1_x = get_bp_array('spine_1', is_x=True)
    spine1_y = get_bp_array('spine_1', is_x=False)
    spine2_x = get_bp_array('spine_2', is_x=True)
    spine2_y = get_bp_array('spine_2', is_x=False)
    hip_left_x = get_bp_array('hip_left', is_x=True)
    hip_left_y = get_bp_array('hip_left', is_x=False)
    hip_right_x = get_bp_array('hip_right', is_x=True)
    hip_right_y = get_bp_array('hip_right', is_x=False)
    neck_x = get_bp_array('neck', is_x=True)  # Reuse
    neck_y = get_bp_array('neck', is_x=False)
    
    avg_body_x = np.nanmean(np.stack([spine1_x, spine2_x, hip_left_x, hip_right_x, neck_x], axis=-1), axis=-1)
    avg_body_y = np.nanmean(np.stack([spine1_y, spine2_y, hip_left_y, hip_right_y, neck_y], axis=-1), axis=-1)
    
    tail_base_x = keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 0]
    tail_base_y = keypoints[:, :, MASTER_SKELETON_MAP['tail_base'], 1]
    head_center_x = keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 0]  # Updated now
    head_center_y = keypoints[:, :, MASTER_SKELETON_MAP['head_center'], 1]
    fallback_x = (head_center_x + tail_base_x) / 2
    fallback_y = (head_center_y + tail_base_y) / 2
    fallback_x[np.isnan(fallback_x)] = np.nan  # If either missing
    fallback_y[np.isnan(fallback_y)] = np.nan
    
    final_body_x = np.where(~np.isnan(body_center_x), body_center_x,
                   np.where(~np.isnan(avg_body_x), avg_body_x, fallback_x))
    final_body_y = np.where(~np.isnan(body_center_y), body_center_y,
                   np.where(~np.isnan(avg_body_y), avg_body_y, fallback_y))
    keypoints[:, :, MASTER_SKELETON_MAP['body_center'], 0] = final_body_x
    keypoints[:, :, MASTER_SKELETON_MAP['body_center'], 1] = final_body_y
    
    # Create validity mask before imputation (1: valid/original non-NaN, 0: invalid/NaN)
    mask = (~np.isnan(keypoints[..., 0])).astype(np.uint8)  # [frames, mice, bp] since x/y same

    behavior_mask = np.zeros((max_pairs, num_actions), dtype=bool)
    try:
        behaviors_labeled = ast.literal_eval(row['behaviors_labeled'])

        mouse_to_idx = {'mouse1': 0, 'mouse2': 1, 'mouse3': 2, 'mouse4': 3}
        for behavior_str in behaviors_labeled:
            parts = behavior_str.split(',')
            if len(parts) != 3:
                continue

            agent, target, action = parts
            agent_idx = mouse_to_idx.get(agent, -1)
            if target == 'self':
                target_idx = agent_idx
            else:
                target_idx = mouse_to_idx.get(target, -1)

            if agent_idx == -1 or target_idx == -1:
                continue

            pair_idx = agent_idx * 4 + target_idx
            action_idx = cfg.action_id_map.get(action, -1)
            if action_idx == -1:
                continue

            behavior_mask[pair_idx, action_idx] = True

        no_action_idx = cfg.action_id_map.get('no_action', -1)
        if no_action_idx != -1:
            behavior_mask[:, no_action_idx] = True  

    except Exception as e:
        print(f"Warning!: Could not parse behaviors_labeled for {video_id}: {e}")

    np.save(os.path.join(PREPROC_DIR, split, f'{video_id}_behavior_mask.npy'), behavior_mask)
    
    # Impute NaNs per trajectory (loop is fine, small iterations)
    # for m in range(max_mice):
    #     for b in range(NUM_MASTER_KEYPOINTS):
    #         for c in range(2):
    #             traj = keypoints[:, m, b, c]
    #             nan_mask = np.isnan(traj)
    #             if np.all(nan_mask):
    #                 traj[:] = np.nan #0.0    instead of 0.0 we keep nans and fix after feature creation.
    #             else:
    #                 non_nan_idx = np.flatnonzero(~nan_mask)
    #                 if len(non_nan_idx) > 1:
    #                     interp_func = interp1d(non_nan_idx, traj[~nan_mask], kind='linear', fill_value='extrapolate')
    #                     traj[nan_mask] = interp_func(np.flatnonzero(nan_mask))
    #                 elif len(non_nan_idx) == 1:
    #                     traj[nan_mask] = traj[non_nan_idx[0]]
    
    # Normalize by scale
    keypoints /= pix_per_cm
    
    # Save as float32
    save_path = os.path.join(PREPROC_DIR, split, f'{video_id}_features.npy')
    np.save(save_path, keypoints.astype(np.float32))
    
    # Save mask [frames, mice, bp]
    mask_path = os.path.join(PREPROC_DIR, split, f'{video_id}_mask.npy')
    np.save(mask_path, mask)
    
    # Update shared dict
    if split not in frame_counts_shared:
        frame_counts_shared[split] = {}
    frame_counts_shared[split][video_id] = num_frames
    print(f"Processed {video_id}: {num_frames} frames")

# Process tracking wrapper for pool
def process_tracking_wrapper(args):
    row, split, frame_counts_shared = args
    process_tracking_video(row, split, frame_counts_shared)

# Step: Process Annotations (Train Only, copied from preprocess_2.py)
def process_annotation_video(lab_id, video_id, frame_counts_shared):
    path = os.path.join(DATA_DIR, 'train_annotation', lab_id, f'{video_id}.parquet')
    if not os.path.exists(path):
        print(f"Warning: No annotation for {video_id}")
        return
    num_frames = frame_counts_shared['train'].get(video_id, 0)
    if num_frames == 0:
        print(f"Skipping labels for {video_id}: 0 frames processed (likely tracking file issue)")
        return
    df = pd.read_parquet(path)
    
    # Initialize labels: [frames, pairs, classes]
    # Start all frames for all pairs as 'no_action'
    labels = np.zeros((num_frames, max_pairs, num_actions), dtype=np.float32)
    labels[:, :, NO_ACTION_IDX] = 1.0
    
    # Use itertuples for faster iteration (instead of iterrows)
    for row in df.itertuples(index=False):
        try:
            agent = int(row.agent_id) if str(row.agent_id).isdigit() else row.agent_id
            target = int(row.target_id) if str(row.target_id).isdigit() else row.target_id
            
            agent_idx = mouse_id_map.get(agent, -1)
            target_idx = mouse_id_map.get(target, -1)
            
            if agent_idx < 0 or target_idx < 0:
                continue
            
            pair_idx = agent_idx * max_mice + target_idx
            
            act = row.action
            act_idx = action_map.get(act, NO_ACTION_IDX)  # Directly default to NO_ACTION_IDX
            
            start = max(0, int(row.start_frame))
            stop = min(num_frames, int(row.stop_frame))
            
            if start >= stop:
                continue
            # When we set a behavior, clear no_action and set the behavior
            if act_idx != NO_ACTION_IDX:
                labels[start:stop, pair_idx, NO_ACTION_IDX] = 0.0
                labels[start:stop, pair_idx, act_idx] = 1.0
            # No need for else, as default is no_action
        
        except Exception as e:
            print(f"Error processing annotation row in {video_id}: {e} | Row: {row}")
    
    save_path = os.path.join(PREPROC_DIR, 'train', f'{video_id}_labels.npy')
    np.save(save_path, labels)
    print(f"Processed labels for {video_id}")

# Annotation wrapper
def process_annotation_wrapper(args):
    lab_id, video_id, frame_counts_shared = args
    process_annotation_video(lab_id, video_id, frame_counts_shared)

# Main execution
if __name__ == "__main__":
    print(f"Starting preprocessing. Outputting to: {PREPROC_DIR}")
    
    try:
        # train_meta = pd.read_csv(os.path.join(DATA_DIR, 'splits/lab_stratified_train.csv'))
        train_meta = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test_meta = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    except FileNotFoundError:
        print(f"Error: Metadata files not found in {DATA_DIR}")
        print("Please ensure 'train.csv' and 'test.csv' are present.")
        sys.exit(1)
    
    # Build lab_bodyparts_global (optional, for warnings)
    all_meta = pd.concat([train_meta, test_meta], ignore_index=True)
    lab_bodyparts_global = {}
    for lab in all_meta['lab_id'].unique():
        lab_df = all_meta[all_meta['lab_id'] == lab]
        unique_bodyparts = set()
        bodyparts_str = lab_df['body_parts_tracked'].iloc[0]
        if pd.isna(bodyparts_str):
            print(f"Warning: No body_parts_tracked for lab {lab}")
            continue
        bodyparts = bodyparts_str.split(',')
        for part in bodyparts:
            part = part.replace("'", "").replace('"', '').replace('[', '').replace(']', '').strip().lower()
            if part:
                unique_bodyparts.add(part)
        lab_bodyparts_global[lab] = unique_bodyparts
    
    # Shared frame_counts
    manager = Manager()
    frame_counts_shared = manager.dict({'train': manager.dict(), 'test': manager.dict()})
    
    # Process tracking in parallel
    for split, meta in [('train', train_meta), ('test', test_meta)]:
        print(f"--- Processing {split} split ---")
        args_list = [(row.to_dict(), split, frame_counts_shared) for _, row in meta.iterrows()]
        with Pool(processes=os.cpu_count()) as pool:
            list(tqdm(pool.imap(process_tracking_wrapper, args_list), total=len(args_list)))  # Add tqdm for progress
    
    # Process annotations in parallel (train only)
    print("\n--- Processing annotations ---")
    args_list = [(lab_id, video_id, frame_counts_shared) for lab_id, video_id in train_meta[['lab_id', 'video_id']].values]
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_annotation_wrapper, args_list), total=len(args_list)))  # Add tqdm
    
    # Convert shared dict to regular for saving
    frame_counts = {'train': dict(frame_counts_shared['train']), 'test': dict(frame_counts_shared['test'])}
    
    # Save auxiliaries
    np.save(os.path.join(PREPROC_DIR, 'frame_counts.npy'), frame_counts)
    with open(os.path.join(PREPROC_DIR, 'action_map.txt'), 'w') as f:
        f.write(str(action_map))
    np.save(os.path.join(PREPROC_DIR, 'master_bodyparts.npy'), np.array(MASTER_SKELETON))
    
    print(f"\nPreprocessing complete!")
    print(f"no_action index: {NO_ACTION_IDX}")
    print(f"Output directory: {PREPROC_DIR}")
    print("Ready for training!")