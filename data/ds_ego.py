import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
import random
from tqdm import tqdm
import ast

class KeypointAugmentor:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply_noise(self, keypoints):
        """Add Gaussian noise to keypoints"""
        noise = torch.randn_like(keypoints) * self.cfg.aug_noise_std
        return keypoints + noise

    def apply_rotation(self, keypoints):
        """Apply random rotation around estimated center"""
        valid = ~torch.isnan(keypoints[..., 0])
        if valid.any():
            center_x = keypoints[..., 0][valid].mean()
            center_y = keypoints[..., 1][valid].mean()
        else:
            center_x = center_y = 0.0

        angle_deg = random.uniform(-self.cfg.aug_rotation_max_deg, self.cfg.aug_rotation_max_deg)
        angle_rad = torch.deg2rad(torch.tensor(angle_deg))

        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        dx = keypoints[..., 0] - center_x
        dy = keypoints[..., 1] - center_y

        new_x = center_x + dx * cos_a - dy * sin_a
        new_y = center_y + dx * sin_a + dy * cos_a

        keypoints_out = keypoints.clone()
        keypoints_out[..., 0] = new_x
        keypoints_out[..., 1] = new_y
        return keypoints_out

    def apply_scale(self, keypoints):
        """Apply random scaling around center"""
        scale = random.uniform(self.cfg.aug_scale_min, self.cfg.aug_scale_max)
        
        valid = ~torch.isnan(keypoints[..., 0])
        if valid.any():
            center_x = keypoints[..., 0][valid].mean()
            center_y = keypoints[..., 1][valid].mean()
        else:
            center_x = center_y = 0.0

        keypoints_out = keypoints.clone()
        keypoints_out[..., 0] = center_x + (keypoints[..., 0] - center_x) * scale
        keypoints_out[..., 1] = center_y + (keypoints[..., 1] - center_y) * scale
        return keypoints_out

    def apply_flip(self, keypoints):
        """Horizontal flip (mirror x-coordinates)"""
        valid = ~torch.isnan(keypoints[..., 0])
        if valid.any():
            center_x = keypoints[..., 0][valid].mean()
        else:
            center_x = 0.0

        keypoints_out = keypoints.clone()
        keypoints_out[..., 0] = 2 * center_x - keypoints[..., 0]
        
        ear_left_idx = self.cfg.MASTER_SKELETON_MAP['ear_left']
        ear_right_idx = self.cfg.MASTER_SKELETON_MAP['ear_right']
        keypoints_out[:, :, [ear_left_idx, ear_right_idx]] = keypoints_out[:, :, [ear_right_idx, ear_left_idx]]
        
        return keypoints_out

    def apply_translate(self, keypoints):
        """Small random translation"""
        valid_x = keypoints[..., 0][~torch.isnan(keypoints[..., 0])]
        valid_y = keypoints[..., 1][~torch.isnan(keypoints[..., 1])]
        if len(valid_x) > 0 and len(valid_y) > 0:
            range_x = valid_x.max() - valid_x.min()
            range_y = valid_y.max() - valid_y.min()
            shift_x = random.uniform(-self.cfg.aug_translate_max, self.cfg.aug_translate_max) * range_x
            shift_y = random.uniform(-self.cfg.aug_translate_max, self.cfg.aug_translate_max) * range_y
        else:
            shift_x = shift_y = 0.0

        keypoints_out = keypoints.clone()
        keypoints_out[..., 0] += shift_x
        keypoints_out[..., 1] += shift_y
        return keypoints_out

    def apply_dropout(self, keypoints):
        """Randomly dropout (set to NaN) some keypoints to simulate occlusions"""
        mask = torch.rand_like(keypoints[..., 0]) < self.cfg.aug_dropout_prob
        keypoints_out = keypoints.clone()
        keypoints_out[mask.unsqueeze(-1).expand_as(keypoints)] = float('nan')
        return keypoints_out

    def apply_time_warp(self, keypoints, labels=None):
        """Time warping: stretch/compress random segments"""
        T, M, B, C = keypoints.shape
        warp_factor = random.uniform(1 - self.cfg.aug_time_warp_factor, 1 + self.cfg.aug_time_warp_factor)
        
        orig_t = torch.linspace(0, 1, T)
        warped_t = torch.pow(orig_t, warp_factor)
        warped_t = (warped_t - warped_t.min()) / (warped_t.max() - warped_t.min()) * (T - 1)
        
        keypoints_out = torch.zeros_like(keypoints)
        for m in range(M):
            for b in range(B):
                for c in range(C):
                    valid = ~torch.isnan(keypoints[:, m, b, c])
                    if valid.sum() > 1:
                        interp = torch.interpolate(torch.tensor([keypoints[valid, m, b, c]]), size=(T,), mode='linear')
                        keypoints_out[:, m, b, c] = interp.squeeze(0)
                    else:
                        keypoints_out[:, m, b, c] = keypoints[:, m, b, c]
        
        if labels is not None:
            labels_out = torch.zeros_like(labels)
            for p in range(labels.shape[1]):
                for a in range(labels.shape[2]):
                    interp = torch.interpolate(labels[:, p, a].unsqueeze(0).unsqueeze(0), size=(T,), mode='nearest')
                    labels_out[:, p, a] = interp.squeeze()
            return keypoints_out, labels_out
        return keypoints_out, labels

    def augment(self, keypoints, labels=None):
        """Apply sequence of augmentations"""
        if random.random() > self.cfg.aug_prob:
            return keypoints, labels

        augs = [
            (self.apply_noise, 0.7),
            (self.apply_rotation, 0.4),
            (self.apply_scale, 0.4),
            (self.apply_translate, 0.3),
            (self.apply_dropout, 0.2),
        ]
        if random.random() < self.cfg.aug_flip_prob:
            keypoints = self.apply_flip(keypoints)

        random.shuffle(augs)
        for aug_fn, prob in augs:
            if random.random() < prob:
                if aug_fn == self.apply_time_warp:
                    keypoints, labels = aug_fn(keypoints, labels)
                else:
                    keypoints = aug_fn(keypoints)

        return keypoints, labels


def batch_to_device(batch, device):
    batch_dict = {}
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch_dict[key] = batch[key].to(device)
        else:
            batch_dict[key] = batch[key]
    return batch_dict


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):
        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.aug = aug
        self.augmentor = KeypointAugmentor(cfg) if mode == 'train' else None

        self.base_data_dir = cfg.preprocessing_basedir + '/train'
        valid_videos = []
        self.video_info = {}
        
        print(f"Initializing CustomDataset (mode={mode}). Scanning {len(self.df)} videos...")
        frame_counts = np.load(f'{self.cfg.preprocessing_basedir}/frame_counts.npy', allow_pickle=True).item()

        num_mice = len(cfg.set_mice)
        num_pairs = num_mice * num_mice
        num_actions = len(cfg.set_behavior_classes)
        mouse_to_idx = {'mouse1': 0, 'mouse2': 1, 'mouse3': 2, 'mouse4': 3}

        for idx in tqdm(range(len(self.df)), desc="Scanning videos and pre-loading labels"):
            row = self.df.iloc[idx]
            video_id = row['video_id']
            features_path = f'{self.base_data_dir}/{video_id}_features.npy'
            labels_path = f'{self.base_data_dir}/{video_id}_labels.npy'

            num_frames = frame_counts.get('train', {}).get(video_id, 0)
            behavior_frames = np.array([], dtype=np.int64)
            
            if os.path.exists(features_path) and os.path.exists(labels_path) and num_frames > 0:
                valid_videos.append(idx)
                
                if mode == 'train':
                    try:
                        labels = np.load(labels_path)
                        no_action_idx = cfg.action_id_map.get('no_action', -1)
                        if no_action_idx != -1:
                            # Sum all channels except no_action
                            action_channels = list(range(labels.shape[2]))
                            action_channels.remove(no_action_idx)
                            frame_sums = labels[:, :, action_channels].sum(axis=(1, 2))
                        else:
                            frame_sums = labels.sum(axis=(1, 2))
                        behavior_frames = np.where(frame_sums > 0)[0]
                    except Exception as e:
                        print(f"Warning: Could not load labels for {video_id}: {e}")

                # Compute behavior_mask
                behaviors_labeled = ast.literal_eval(row['behaviors_labeled']) if isinstance(row['behaviors_labeled'], str) else row['behaviors_labeled']
                behavior_mask = np.zeros((num_pairs, num_actions), dtype=bool)
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
                    pair_idx = agent_idx * num_mice + target_idx
                    action_idx = cfg.action_id_map.get(action, -1)
                    if action_idx == -1:
                        continue
                    behavior_mask[pair_idx, action_idx] = True
                
                no_action_idx = cfg.action_id_map.get('no_action', -1)
                if no_action_idx != -1:
                    behavior_mask[:, no_action_idx] = True

                self.video_info[video_id] = {
                    'num_frames': num_frames,
                    'duration_sec': row.get('video_duration_sec', num_frames / 30),
                    'lab_id': row['lab_id'],
                    'behavior_frames': behavior_frames,
                    'behavior_mask': behavior_mask
                }

        self.df = self.df.iloc[valid_videos].reset_index(drop=True)
        print(f"Dataset {mode}: {len(valid_videos)} valid videos out of {len(df)} total")

        # Create sampling schedule
        self.create_sampling_schedule()

    def create_sampling_schedule(self):
        self.sampling_schedule = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            video_id = row['video_id']
            if video_id not in self.video_info:
                continue

            num_frames = self.video_info[video_id]['num_frames']
            if num_frames <= self.cfg.window_size:
                num_windows = 1
            else:
                stride = self.cfg.stride if hasattr(self.cfg, 'stride') else self.cfg.window_size // 2
                total_possible_windows = (num_frames // stride ) + 1

                if self.mode == 'train':
                    num_windows = max(
                        self.cfg.min_windows_per_video,
                        min(
                            self.cfg.max_windows_per_video,
                            int(total_possible_windows * self.cfg.windows_per_epoch_ratio)
                        )
                    )
                else:
                    num_windows = total_possible_windows

            # Add entries to schedule
            for win_idx in range(num_windows):
                self.sampling_schedule.append({
                    'video_id': video_id,
                    'df_idx': idx,
                    'win_idx': win_idx,
                    'num_frames': num_frames
                })

        print(f"Created sampling schedule with {len(self.sampling_schedule)} windows")
        if len(self.df) > 0:
            print(f"Average windows per video: {len(self.sampling_schedule) / len(self.df):.2f}")

    def __len__(self):
        return len(self.sampling_schedule)

    def __getitem__(self, idx):
        sample_info = self.sampling_schedule[idx]
        video_id = sample_info['video_id']
        num_frames = sample_info['num_frames']
        
        video_data = self.video_info[video_id]
        
        # Load data with mmap
        features_mmap = np.load(f'{self.base_data_dir}/{video_id}_features.npy', mmap_mode='r')
        labels_mmap = np.load(f'{self.base_data_dir}/{video_id}_labels.npy', mmap_mode='r') if self.mode != 'test' else None
    
        if self.mode == 'train':
            positive_frames = video_data.get('behavior_frames', np.array([], dtype=np.int64))
            
            if random.random() < self.cfg.bias_prob and len(positive_frames) > 0:
                center_frame = random.choice(positive_frames)
                offset = random.randint(0, self.cfg.window_size - 1)
                start = max(0, min(center_frame - offset, num_frames - self.cfg.window_size))
            else:
                start = random.randint(0, max(0, num_frames - self.cfg.window_size))
        else: 
            stride = self.cfg.stride if hasattr(self.cfg, 'stride') else self.cfg.window_size // 2
            start = sample_info['win_idx'] * stride
            if num_frames <= self.cfg.window_size:
                start = 0

        if start + self.cfg.window_size > num_frames:
            start = num_frames - self.cfg.window_size
        
        if self.cfg.window_size >= num_frames:
            start = 0

        # Extract window
        end = min(start + self.cfg.window_size, num_frames)
        features = torch.from_numpy(features_mmap[start:end].copy()).float()

        if labels_mmap is not None and self.mode == 'train':
            labels = torch.from_numpy(labels_mmap[start:end].copy()).float()
        else:
            labels = None

        if self.mode == 'train' and self.augmentor is not None and self.cfg.augment:
            features, labels = self.augmentor.augment(features, labels)

        # Pad if necessary
        actual_len = features.shape[0]
        if actual_len < self.cfg.window_size:
            pad_len = self.cfg.window_size - actual_len

            features = torch.cat([features, torch.full((pad_len, *features.shape[1:]), float('nan'))], dim=0)
            if labels is not None:
                labels = torch.cat([labels, torch.zeros(pad_len, *labels.shape[1:])], dim=0)
                no_action_idx = self.cfg.action_id_map.get('no_action', 38)
                labels[actual_len:, :, no_action_idx] = 1.0 

        # Create mask
        mask = torch.ones(self.cfg.window_size)
        if actual_len < self.cfg.window_size:
            mask[actual_len:] = 0

        # Apply time reversal if configured
        if self.cfg.reverse_time:
            features = torch.flip(features, dims=[0])
            if labels is not None:
                labels = torch.flip(labels, dims=[0])
            mask = torch.flip(mask, dims=[0])
        
        # Compute features
        features, per_mouse_features = self.compute_feature(features)

        item = {
            'input': features,
            'input_mice': per_mouse_features,
            'input_mask': mask,
            'labels': labels,
            'video_id': video_id,
            'start_frame': torch.tensor(start),
            'num_frames': torch.tensor(num_frames)
        }

        # Add behavior_mask (per video)
        if 'behavior_mask' in video_data:
            item['behavior_mask'] = torch.from_numpy(video_data['behavior_mask']).float()

        return item

    def compute_egocentric_transform(self, keypoints, bp_to_idx):
        """
        Computes Rotation Matrices and Translation vectors for Egocentric alignment.
        Result: Every mouse, at every frame, is centered at (0,0) with spine along +X.
        Returns:
            centers: [T, M, 2] - The translation vector to subtract
            rot_matrices: [T, M, 2, 2] - The rotation matrix to multiply
        """
        T, M, B, C = keypoints.shape
        device = keypoints.device
        
        centers = keypoints[:, :, bp_to_idx['body_center']].clone() # [T, M, 2]
        
        heads = keypoints[:, :, bp_to_idx['head_center']]
        tails = keypoints[:, :, bp_to_idx['tail_base']]
        
      
        spines = heads - tails # [T, M, 2]
        
      
        spines = torch.nan_to_num(spines, 0.0)
        
        # Prevent zero-division
        spine_norms = torch.norm(spines, dim=-1, keepdim=True)
        # If norm is too close to 0, use [1, 0]
        tiny_mask = spine_norms < 1e-6
        spines = torch.where(tiny_mask, torch.tensor([1.0, 0.0], device=device), spines)
        spine_norms = torch.where(tiny_mask, torch.tensor(1.0, device=device), spine_norms)
        

        
        cos_theta = spines[..., 0:1] / spine_norms
        sin_theta = spines[..., 1:2] / spine_norms
        

        row1 = torch.cat([cos_theta, sin_theta], dim=-1)
        row2 = torch.cat([-sin_theta, cos_theta], dim=-1)
        
        rot_matrices = torch.stack([row1, row2], dim=-2)
        
        return centers, rot_matrices

    def compute_feature(self, keypoints):
        T, M, B, C = keypoints.shape
        device = keypoints.device
        
        FPS = 30
        MAX_VELOCITY = 100.0
        MAX_ACCEL = 500.0
        MAX_JERK = 1000.0
        EPSILON = 1e-6
        
        bp_to_idx = self.cfg.MASTER_SKELETON_MAP
        
        # Helper function for safe division
        def safe_divide(a, b, eps=EPSILON):
            return a / (b + eps)
        
        # Helper function for angle computation
        def compute_angle_3points(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            norm_v1 = torch.norm(v1, dim=-1, keepdim=True)
            norm_v2 = torch.norm(v2, dim=-1, keepdim=True)
            cos_angle = torch.sum(v1 * v2, dim=-1) / (norm_v1.squeeze(-1) * norm_v2.squeeze(-1) + EPSILON)
            return torch.acos(torch.clamp(cos_angle, -1.0 + EPSILON, 1.0 - EPSILON))
        
        all_features = []
        
        # ============== KINEMATICS (GLOBAL) ==============
        velocities = torch.diff(keypoints, dim=0, prepend=keypoints[0:1]) * FPS
        velocities = torch.clamp(velocities, -MAX_VELOCITY, MAX_VELOCITY)
        accelerations = torch.diff(velocities, dim=0, prepend=velocities[0:1])
        accelerations = torch.clamp(accelerations, -MAX_ACCEL, MAX_ACCEL)
        jerks = torch.diff(accelerations, dim=0, prepend=accelerations[0:1])
        jerks = torch.clamp(jerks, -MAX_JERK, MAX_JERK)
        
        speeds_normalized = torch.norm(velocities, dim=-1) / MAX_VELOCITY
        accel_normalized = torch.norm(accelerations, dim=-1) / MAX_ACCEL
        jerk_normalized = torch.norm(jerks, dim=-1) / MAX_JERK
        
        all_features.append(speeds_normalized.reshape(T, M * B))
        all_features.append(accel_normalized.reshape(T, M * B))
        all_features.append(jerk_normalized.reshape(T, M * B))
        
        # Angular velocities
        angular_velocities = torch.zeros(T, M, B).to(device)
        for m in range(M):
            for b in range(B):
                vx, vy = velocities[:, m, b, 0], velocities[:, m, b, 1]
                angles = torch.atan2(vy, vx)
                angular_velocities[1:, m, b] = torch.diff(angles)
                angular_velocities[:, m, b] = (angular_velocities[:, m, b] + torch.pi) % (2 * torch.pi) - torch.pi
        
        angular_velocities_normalized = angular_velocities / torch.pi
        all_features.append(angular_velocities_normalized.reshape(T, M * B))
        
        # ============== CENTROID & SHAPE ==============
        centroid_parts = ['head_center', 'body_center', 'tail_base']
        centroid_indices = torch.tensor([bp_to_idx[bp] for bp in centroid_parts]).to(device)
        mouse_centroids = torch.mean(keypoints[:, :, centroid_indices, :], dim=2)
        
        centroid_velocity = torch.diff(mouse_centroids, dim=0, prepend=mouse_centroids[0:1]) * FPS
        centroid_velocity = torch.clamp(centroid_velocity, -MAX_VELOCITY, MAX_VELOCITY)
        centroid_speed_normalized = torch.norm(centroid_velocity, dim=-1) / MAX_VELOCITY
        
        centroid_accel = torch.diff(centroid_velocity, dim=0, prepend=centroid_velocity[0:1])
        centroid_accel_normalized = torch.norm(torch.clamp(centroid_accel, -MAX_ACCEL, MAX_ACCEL), dim=-1) / MAX_ACCEL
        
        all_features.extend([centroid_speed_normalized, centroid_accel_normalized])
        
        movement_heading = torch.atan2(centroid_velocity[:, :, 1], centroid_velocity[:, :, 0])
        movement_heading_normalized = movement_heading / torch.pi
        all_features.append(movement_heading_normalized)
        
        # Shape / Body Configuration
        nose_pos = keypoints[:, :, bp_to_idx['nose'], :]
        tail_base_pos = keypoints[:, :, bp_to_idx['tail_base'], :]
        head_center_pos = keypoints[:, :, bp_to_idx['head_center'], :]
        body_center_pos = keypoints[:, :, bp_to_idx['body_center'], :]
        ear_left_pos = keypoints[:, :, bp_to_idx['ear_left'], :]
        ear_right_pos = keypoints[:, :, bp_to_idx['ear_right'], :]
        
        body_length = torch.norm(nose_pos - tail_base_pos, dim=-1)
        body_length_95 = torch.quantile(body_length[~torch.isnan(body_length)], 0.95) if body_length[~torch.isnan(body_length)].numel() > 0 else 1.0
        body_length_normalized = torch.clamp(body_length / (body_length_95 + EPSILON), 0, 2)
        all_features.append(body_length_normalized)
        
        body_length_change = torch.diff(body_length, dim=0, prepend=body_length[0:1])
        body_length_change_normalized = torch.clamp(body_length_change / (body_length_95 * 0.1 + EPSILON), -2, 2)
        all_features.append(body_length_change_normalized)
        
        ear_spread = torch.norm(ear_left_pos - ear_right_pos, dim=-1)
        ear_spread_95 = torch.quantile(ear_spread[~torch.isnan(ear_spread)], 0.95) if ear_spread[~torch.isnan(ear_spread)].numel() > 0 else 1.0
        ear_spread_normalized = torch.clamp(ear_spread / (ear_spread_95 + EPSILON), 0, 2)
        all_features.append(ear_spread_normalized)
        
        body_elongation = safe_divide(body_length, ear_spread)
        body_elongation_normalized = torch.clamp(body_elongation, 0, 10) / 10
        all_features.append(body_elongation_normalized)
        
        # Angles
        body_curvature = compute_angle_3points(nose_pos, body_center_pos, tail_base_pos)
        body_curvature_normalized = body_curvature / torch.pi
        all_features.append(body_curvature_normalized)
        
        head_angle = compute_angle_3points(nose_pos, head_center_pos, body_center_pos)
        head_angle_normalized = head_angle / torch.pi
        all_features.append(head_angle_normalized)
        
        tail_angle = compute_angle_3points(head_center_pos, body_center_pos, tail_base_pos)
        tail_angle_normalized = tail_angle / torch.pi
        all_features.append(tail_angle_normalized)
        
        body_vector = nose_pos - tail_base_pos
        body_orientation = torch.atan2(body_vector[:, :, 1], body_vector[:, :, 0])
        body_orientation_normalized = body_orientation / torch.pi
        all_features.append(body_orientation_normalized)
        
        unwrapped_orientation = torch.from_numpy(np.unwrap(body_orientation.cpu().numpy(), axis=0)).to(device)
        body_angular_velocity = torch.diff(unwrapped_orientation, dim=0, prepend=unwrapped_orientation[0:1])
        body_angular_velocity = torch.clamp(body_angular_velocity, -torch.pi, torch.pi) / torch.pi
        all_features.append(body_angular_velocity)
        
        nose_centroid_dist = torch.norm(nose_pos - mouse_centroids, dim=-1)
        nose_centroid_normalized = torch.clamp(safe_divide(nose_centroid_dist, body_length), 0, 2)
        tail_centroid_dist = torch.norm(tail_base_pos - mouse_centroids, dim=-1)
        tail_centroid_normalized = torch.clamp(safe_divide(tail_centroid_dist, body_length), 0, 2)
        all_features.extend([nose_centroid_normalized, tail_centroid_normalized])
        
        nose_velocity = velocities[:, :, bp_to_idx['nose'], :]
        tail_velocity = velocities[:, :, bp_to_idx['tail_base'], :]
        nose_rel_speed = torch.norm(nose_velocity - centroid_velocity, dim=-1) / MAX_VELOCITY
        tail_rel_speed = torch.norm(tail_velocity - centroid_velocity, dim=-1) / MAX_VELOCITY
        all_features.extend([nose_rel_speed, tail_rel_speed])
        
        # ============== EGOCENTRIC ALIGNMENT (THE FIX) ==============
        ego_centers, ego_rot_mats = self.compute_egocentric_transform(keypoints, bp_to_idx)
        
        typical_mouse_size = torch.median(body_length[~torch.isnan(body_length)]) if body_length[~torch.isnan(body_length)].numel() > 0 else 1.0

        # ============== INTER-MOUSE FEATURES ==============
        inter_features_list = []
        
        # Per-mouse lists
        mouse_features = [[] for _ in range(M)]
        
        # 1. Add Intra-mouse features
        bp_specific = [speeds_normalized, accel_normalized, jerk_normalized, angular_velocities_normalized]
        for feat in bp_specific:
            for m in range(M):
                mouse_features[m].append(feat[:, m, :].reshape(T, B))
        
        mouse_specific = [
            centroid_speed_normalized, centroid_accel_normalized, movement_heading_normalized,
            body_length_normalized, body_length_change_normalized, ear_spread_normalized, body_elongation_normalized,
            body_curvature_normalized, head_angle_normalized, tail_angle_normalized, body_orientation_normalized, body_angular_velocity,
            nose_centroid_normalized, tail_centroid_normalized, nose_rel_speed, tail_rel_speed
        ]
        for feat in mouse_specific:
            for m in range(M):
                mouse_features[m].append(feat[:, m].unsqueeze(-1))
        
        # 2. Add Egocentric Interaction Features
        for i in range(M): # Agent
            R_i = ego_rot_mats[:, i] # [T, 2, 2]
            Center_i = ego_centers[:, i] # [T, 2]
            
            for j in range(M): # Target
                if i == j:
                    continue
                
                centroid_i = mouse_centroids[:, i, :]
                centroid_j = mouse_centroids[:, j, :]
                dist_centroids = torch.norm(centroid_j - centroid_i, dim=-1)
                dist_centroids_normalized = torch.clamp(dist_centroids / (typical_mouse_size * 5 + EPSILON), 0, 2)
                mouse_features[i].append(dist_centroids_normalized.unsqueeze(-1))
                inter_features_list.append(dist_centroids_normalized.unsqueeze(-1))
                
                dist_change = torch.diff(dist_centroids, dim=0, prepend=dist_centroids[0:1])
                dist_change_normalized = torch.clamp(dist_change / (typical_mouse_size + EPSILON), -2, 2) / 2
                mouse_features[i].append(dist_change_normalized.unsqueeze(-1))
                inter_features_list.append(dist_change_normalized.unsqueeze(-1))
                
                nose_i = nose_pos[:, i, :]
                nose_j = nose_pos[:, j, :]
                tail_j = tail_base_pos[:, j, :]
                
                nose_nose_dist = torch.norm(nose_i - nose_j, dim=-1)
                nose_nose_normalized = torch.clamp(nose_nose_dist / (typical_mouse_size * 3 + EPSILON), 0, 2)
                mouse_features[i].append(nose_nose_normalized.unsqueeze(-1))
                inter_features_list.append(nose_nose_normalized.unsqueeze(-1))
                
                nose_tail_dist = torch.norm(nose_i - tail_j, dim=-1)
                nose_tail_normalized = torch.clamp(nose_tail_dist / (typical_mouse_size * 3 + EPSILON), 0, 2)
                mouse_features[i].append(nose_tail_normalized.unsqueeze(-1))
                inter_features_list.append(nose_tail_normalized.unsqueeze(-1))
                
                rel_speed = (centroid_speed_normalized[:, i] - centroid_speed_normalized[:, j])
                mouse_features[i].append(rel_speed.unsqueeze(-1))
                inter_features_list.append(rel_speed.unsqueeze(-1))
                
                vec_to_j = centroid_j - centroid_i
                heading_vec_i = body_vector[:, i, :]
                dot_prod = torch.sum(heading_vec_i * vec_to_j, dim=-1)
                norm_prod = torch.norm(heading_vec_i, dim=-1) * torch.norm(vec_to_j, dim=-1)
                approach_angle = torch.acos(torch.clamp(safe_divide(dot_prod, norm_prod), -1.0 + EPSILON, 1.0 - EPSILON))
                approach_angle_normalized = approach_angle / torch.pi
                mouse_features[i].append(approach_angle_normalized.unsqueeze(-1))
                inter_features_list.append(approach_angle_normalized.unsqueeze(-1))
                
             
                target_pts = {
                    'nose': nose_pos[:, j, :],
                    'body': mouse_centroids[:, j, :],
                    'tail': tail_base_pos[:, j, :]
                }
                
                for pt_name, pt_tensor in target_pts.items():
                    rel_vec = pt_tensor - Center_i # [T, 2]
                    
                  
                    rel_vec_ego = torch.matmul(R_i, rel_vec.unsqueeze(-1)).squeeze(-1) # [T, 2]
                    
                    norm_scale = typical_mouse_size * 5.0 + EPSILON
                    
                    ego_x = torch.clamp(rel_vec_ego[:, 0] / norm_scale, -2.0, 2.0)
                    ego_y = torch.clamp(rel_vec_ego[:, 1] / norm_scale, -2.0, 2.0)
                    
                    mouse_features[i].append(ego_x.unsqueeze(-1))
                    mouse_features[i].append(ego_y.unsqueeze(-1))
                    
                    inter_features_list.append(ego_x.unsqueeze(-1))
                    inter_features_list.append(ego_y.unsqueeze(-1))

        if inter_features_list:
            inter_features = torch.cat(inter_features_list, dim=-1)
            all_features.append(inter_features)
        
        per_mouse_feats = [torch.cat(feats, dim=-1) for feats in mouse_features]
        
        features = torch.cat(all_features, dim=-1)
        per_mouse_feats = torch.stack(per_mouse_feats, dim=1)

        features = torch.clamp(features, -10, 10)
        per_mouse_feats = torch.clamp(per_mouse_feats, -10, 10)
        
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        per_mouse_feats = torch.nan_to_num(per_mouse_feats, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return features, per_mouse_feats

def collate_fn(batch):
    """Custom collate function for batching"""
    inputs = torch.stack([item['input'] for item in batch])
    inputs_mice = torch.stack([item['input_mice'] for item in batch]) if 'input_mice' in batch[0] else None
    masks = torch.stack([item['input_mask'] for item in batch])
    behavior_masks = torch.stack([item['behavior_mask'] for item in batch]) if 'behavior_mask' in batch[0] else None

    labels = torch.stack([item['labels'] for item in batch]) if batch[0]['labels'] is not None else None

    video_ids = [item['video_id'] for item in batch]
    start_frames = torch.stack([item['start_frame'] for item in batch])
    num_frames = torch.stack([item['num_frames'] for item in batch])

    collated = {
        'input': inputs,
        'input_mice': inputs_mice,
        'input_mask': masks,
        'labels': labels,
        'video_id': video_ids,
        'start_frame': start_frames,
        'num_frames': num_frames
    }
    if behavior_masks is not None:
        collated['behavior_mask'] = behavior_masks

    return collated
    
tr_collate_fn = collate_fn
val_collate_fn = collate_fn