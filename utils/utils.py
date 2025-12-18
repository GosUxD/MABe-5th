import random
import os
import numpy as np
import torch
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, median_filter


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # Use a global seed + worker_id to create a unique seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def process_one_video_preds_vectorized(cfg, video_preds, video_id, lab_id, num_frames,
                                       mouse_map, action_map, num_pairs, num_actions,
                                       min_duration):
    # video_preds already has softmax applied from model
    # Shape: [num_frames, num_pairs, num_actions]

    # Get the best action for each frame and pair
    best_action_indices = np.argmax(video_preds, axis=2)  # Shape [num_frames, num_pairs]
    video_predictions = []

    # Get number of mice
    num_mice = int(np.sqrt(num_pairs))

    for pair_idx in range(num_pairs):
        agent_idx = pair_idx // num_mice
        target_idx = pair_idx % num_mice

        agent_id = mouse_map[agent_idx]
        target_id = mouse_map[target_idx] if agent_idx != target_idx else 'self'

        for action_idx in range(num_actions):
            action = action_map[action_idx]
            if action == 'no_action':
                continue

            # Skip invalid combinations based on detected action types
            if agent_idx == target_idx and action in cfg.social_only_actions:
                continue  # Can't do social actions on self
            if agent_idx != target_idx and action in cfg.self_only_actions:
                continue  # Can't do self actions on others

            # Boolean mask for this action
            behavior_frames = (best_action_indices[:, pair_idx] == action_idx)

            # Vectorized way to find contiguous True segments
            padded = np.concatenate(([False], behavior_frames, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0] - 1  # Adjust for exclusive end

            # Filter by min_duration
            durations = ends - starts + 1
            valid = durations >= min_duration
            for start, end in zip(starts[valid], ends[valid]):
                video_predictions.append({
                    'video_id': video_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': action,
                    'start_frame': start,
                    'stop_frame': end,
                })

    if video_predictions:
        return pd.DataFrame(video_predictions)
    return None


def process_one_video_preds_filters(cfg, video_preds, video_id, lab_id, num_frames,
                                       mouse_map, action_map, num_pairs, num_actions,
                                       min_duration, smooth_sigma=0.0, conf_thresh=0.0,
                                       max_gap=0, use_median_filter=False):

    # video_preds: [num_frames, num_pairs, num_actions] softmax probs

    # Step 1: Temporal smoothing of probabilities (per pair/action)
    if smooth_sigma > 0:
        smoothed_preds = np.zeros_like(video_preds)
        for pair in range(num_pairs):
            for act in range(num_actions):
                smoothed_preds[:, pair, act] = gaussian_filter1d(video_preds[:, pair, act], sigma=smooth_sigma)
        video_preds = smoothed_preds / np.sum(smoothed_preds, axis=2, keepdims=True)  # Re-normalize

    # Get best actions
    best_action_indices = np.argmax(video_preds, axis=2)  # [num_frames, num_pairs]
    best_probs = np.max(video_preds, axis=2)  # [num_frames, num_pairs] for confidence

    # Optional: Median filter on labels (after argmax)
    if use_median_filter:
        for pair in range(num_pairs):
            best_action_indices[:, pair] = median_filter(best_action_indices[:, pair], size=5)  # Window size tunable

    video_predictions = []
    num_mice = int(np.sqrt(num_pairs))

    for pair_idx in range(num_pairs):
        agent_idx = pair_idx // num_mice
        target_idx = pair_idx % num_mice
        agent_id = mouse_map[agent_idx]
        target_id = mouse_map[target_idx] if agent_idx != target_idx else 'self'

        for action_idx in range(num_actions):
            action = action_map[action_idx]
            if action == 'no_action':
                continue
            if (agent_idx == target_idx and action in cfg.social_only_actions) or \
               (agent_idx != target_idx and action in cfg.self_only_actions):
                continue

            behavior_frames = (best_action_indices[:, pair_idx] == action_idx)

            # Find contiguous segments
            padded = np.concatenate(([False], behavior_frames, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0] - 1

            # Step 2: Merge gaps (for same action)
            if len(starts) > 1 and max_gap > 0:
                merged_starts, merged_ends = [starts[0]], [ends[0]]
                for i in range(1, len(starts)):
                    if starts[i] - merged_ends[-1] - 1 <= max_gap:
                        merged_ends[-1] = ends[i]  # Merge
                    else:
                        merged_starts.append(starts[i])
                        merged_ends.append(ends[i])
                starts, ends = np.array(merged_starts), np.array(merged_ends)

            # Filter by min_duration and confidence
            durations = ends - starts + 1
            valid = durations >= min_duration
            for start, end in zip(starts[valid], ends[valid]):
                seg_probs = video_preds[start:end+1, pair_idx, action_idx]  # Use action-specific probs
                mean_conf = np.mean(seg_probs)
                if mean_conf < conf_thresh:
                    continue
                video_predictions.append({
                    'video_id': video_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'action': action,
                    'start_frame': start,
                    'stop_frame': end,
                    'confidence': mean_conf
                })

    # Guard: Check and resolve any overlaps per agent-target pair
    if video_predictions:
        df = pd.DataFrame(video_predictions)
        resolved_predictions = []
        for (agent_id, target_id), group in df.groupby(['agent_id', 'target_id']):
            segments = group.to_dict('records')
            segments.sort(key=lambda x: x['start_frame'])
            # Check for overlap
            overlap_found = False
            for k in range(1, len(segments)):
                if segments[k]['start_frame'] <= segments[k-1]['stop_frame']:
                    overlap_found = True
                    break
            if overlap_found:
                # NMS: Sort by descending confidence, greedily select non-overlapping
                segments.sort(key=lambda x: -x['confidence'])
                selected = []
                for seg in segments:
                    if all(max(seg['start_frame'], sel['start_frame']) > min(seg['stop_frame'], sel['stop_frame']) for sel in selected):
                        selected.append(seg)
                resolved_predictions.extend(selected)
            else:
                resolved_predictions.extend(segments)
        if resolved_predictions:
            return pd.DataFrame(resolved_predictions)
    return None


def process_one_video_preds_multiclass(cfg, video_preds, video_id, lab_id, num_frames,
                                       mouse_map, action_map, num_pairs, num_actions,
                                       min_duration, behavior_mask):
    no_action_idx = cfg.action_id_map['no_action']
    # Shape: [num_frames, num_pairs, num_actions]
    best_action_indices = np.argmax(video_preds, axis=2)  # Shape [num_frames, num_pairs]
    video_predictions = []

    # Get number of mice
    num_mice = int(np.sqrt(num_pairs))

    # Get max_gap from cfg, default to 0 (no merging)
    max_gap = cfg.max_gap if hasattr(cfg, 'max_gap') else 20
    min_duration = cfg.min_duration if hasattr(cfg, 'min_duration') else 5

    for pair_idx in range(num_pairs):
        agent_idx = pair_idx // num_mice
        target_idx = pair_idx % num_mice

        agent_id = mouse_map[agent_idx]
        target_id = mouse_map[target_idx] if agent_idx != target_idx else 'self'

        for action_idx in range(num_actions):
            action = action_map[action_idx]
            if action == 'no_action':
                continue

            if not behavior_mask[pair_idx, action_idx]:
                continue

            # # Skip invalid combinations based on detected action types
            # if agent_idx == target_idx and action in cfg.social_only_actions:
            #     continue  # Can't do social actions on self
            # if agent_idx != target_idx and action in cfg.self_only_actions:
            #     continue  # Can't do self actions on others

            # Find frames where this action is predicted with high confidence
            behavior_frames = (best_action_indices[:, pair_idx] == action_idx)

            # Collect initial contiguous segments
            segments = []
            in_behavior = False
            start_frame = 0

            for frame in range(num_frames):
                if behavior_frames[frame] and not in_behavior:
                    start_frame = frame
                    in_behavior = True
                elif not behavior_frames[frame] and in_behavior:
                    stop_frame = frame - 1
                    segments.append([start_frame, stop_frame])
                    in_behavior = False

            # Handle behavior extending to end
            if in_behavior:
                stop_frame = num_frames - 1
                segments.append([start_frame, stop_frame])

            # Merge segments with small gaps
            if segments:
                merged_segments = [segments[0]]
                for seg in segments[1:]:
                    prev_stop = merged_segments[-1][1]
                    gap_start = prev_stop + 1
                    gap_end = seg[0] - 1
                    gap_size = gap_end - gap_start + 1

                    if gap_size > 0 and gap_size <= max_gap:
                        # Check if all gap frames are 'no_action'
                        is_no_action_gap = all(
                            best_action_indices[f, pair_idx] == no_action_idx
                            for f in range(gap_start, gap_end + 1)
                        )
                        if is_no_action_gap:
                            # Merge: extend previous segment to cover this one
                            merged_segments[-1][1] = seg[1]
                            continue

                    # No merge: add as new segment
                    merged_segments.append(seg)
            else:
                merged_segments = []

            # Filter merged segments by min_duration and collect predictions
            for start, stop in merged_segments:
                if stop - start + 1 >= min_duration:
                    video_predictions.append({
                        'video_id': video_id,
                        'agent_id': agent_id,
                        'target_id': target_id,
                        'action': action,
                        'start_frame': start,
                        'stop_frame': stop,
                    })

    if video_predictions:
        return pd.DataFrame(video_predictions)
    return None

def batch_to_device(batch, device):
    batch_dict = {}
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch_dict[key] = batch[key].to(device)
        else:
            batch_dict[key] = batch[key]
    return batch_dict


@torch.no_grad()
def custom_update_bn(
    loader,
    model,
    device=None,
    batch_to_device_fn=batch_to_device,
):
    r"""Adapted from torch.optim.swa_utils.update_bn to handle batch dictionaries.
    
    Performs one pass over data in `loader` to update BatchNorm running statistics in the model.
    Uses `batch_to_device_fn` to move the batch dict to the device before calling model.forward(batch).
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for batch in loader:
        if batch_to_device_fn is not None and device is not None:
            batch = batch_to_device_fn(batch, device)
        model(batch)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class BNUpdateWrapper:
    def __init__(self, dataloader, key='input'):
        self.dataloader = dataloader
        self.key = key

    def __iter__(self):
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        return len(self.dataloader)

def calculate_iou_1d_vectorized(starts, ends, s2, e2):
    """
    Vectorized 1D IoU calculation.
    starts, ends: arrays of shapes (N,)
    s2, e2: scalars
    """
    intersection_min = np.maximum(starts, s2)
    intersection_max = np.minimum(ends, e2)
    intersection = np.maximum(0, intersection_max - intersection_min)
    
    union = (ends - starts) + (e2 - s2) - intersection
    # Avoid division by zero
    union[union <= 0] = 1e-6
    return intersection / union

def fast_soft_nms(proposals, sigma=0.5, iou_thresh=0.1, score_thresh=0.001):
    """
    Optimized Soft-NMS. 
    Expects 'proposals' to be a list of lists OR a numpy array: 
    [start, end, score, action_idx, agent_idx, target_idx]
    """
    # 1. FIX: Use len() instead of implicit boolean check
    if len(proposals) == 0:
        return []
        
    # 2. Handle input types (list vs array)
    if isinstance(proposals, np.ndarray):
        props = proposals.copy()
    else:
        props = np.array(proposals)
    
    # Sort by score (column 2) descending
    order = props[:, 2].argsort()[::-1]
    props = props[order]
    
    keep = []
    
    while len(props) > 0:
        # Pick best
        best = props[0]
        keep.append(best)
        
        # If only one left, break
        if len(props) == 1:
            break
            
        # Compare best against all others
        # best[0]=start, best[1]=end
        ious = calculate_iou_1d_vectorized(props[1:, 0], props[1:, 1], best[0], best[1])
        
        # Soft-NMS Decay
        # Only decay items with IoU > thresh
        decay_mask = ious > iou_thresh
        weights = np.ones_like(ious)
        weights[decay_mask] = np.exp(-(ious[decay_mask]**2) / sigma)
        
        # Apply weights to scores (column 2)
        props[1:, 2] *= weights
        
        # Filter by score threshold
        mask = props[1:, 2] > score_thresh
        props = props[1:][mask]
        
        # Re-sort (critical for Soft-NMS)
        if len(props) > 0:
            order = props[:, 2].argsort()[::-1]
            props = props[order]
            
    return keep
