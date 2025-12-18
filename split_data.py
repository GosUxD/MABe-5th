"""
Data splitting utilities for MABe Challenge
Provides multiple splitting strategies for train/validation split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, KFold
import os
import json
from collections import defaultdict


class MABeDataSplitter:
    """
    Handles different splitting strategies for MABe mouse behavior detection dataset
    """

    def __init__(self, csv_path='datamount/train_fixed.csv', seed=42):
        self.df = pd.read_csv(csv_path)
        self.seed = seed
        np.random.seed(seed)

        # Analyze data distribution
        self.analyze_data()

    def analyze_data(self):
        """Analyze data distribution across labs and other metadata"""
        print("\n=== Dataset Analysis ===")
        print(f"Total videos: {len(self.df)}")
        print(f"Unique labs: {self.df['lab_id'].nunique()}")

        # Lab distribution
        lab_counts = self.df['lab_id'].value_counts()
        print("\nVideos per lab:")
        for lab, count in lab_counts.items():
            print(f"  {lab}: {count} ({count/len(self.df)*100:.1f}%)")

        # Store metadata for splitting decisions
        self.lab_counts = lab_counts
        self.labs = self.df['lab_id'].unique()

    def lab_stratified_split(self, val_ratio=0.15, min_videos_per_lab=2):
        """
        Stratified split ensuring each lab is represented in both train and val

        Args:
            val_ratio: Proportion of data for validation
            min_videos_per_lab: Minimum videos required for a lab to be split

        Returns:
            train_df, val_df
        """
        print("\n=== Lab-Stratified Split ===")
        print(f"Target validation ratio: {val_ratio}")

        train_indices = []
        val_indices = []

        for lab in self.labs:
            lab_df = self.df[self.df['lab_id'] == lab]
            lab_indices = lab_df.index.tolist()

            if len(lab_df) < min_videos_per_lab:
                # Too few videos to split - add all to train
                train_indices.extend(lab_indices)
                print(f"  {lab}: {len(lab_df)} videos - all to train (too few to split)")
            else:
                # Stratify by additional metadata if available
                n_val = max(1, int(len(lab_df) * val_ratio))
                n_train = len(lab_df) - n_val

                # Shuffle and split
                np.random.shuffle(lab_indices)
                train_indices.extend(lab_indices[:n_train])
                val_indices.extend(lab_indices[n_train:])
                print(f"  {lab}: {n_train} train, {n_val} val")

        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)

        print(f"\nFinal split: {len(train_df)} train ({len(train_df)/len(self.df)*100:.1f}%), "
              f"{len(val_df)} val ({len(val_df)/len(self.df)*100:.1f}%)")

        return train_df, val_df

    def lab_holdout_split(self, holdout_labs=None, n_holdout_labs=3):
        """
        Hold out entire labs for validation to test generalization

        Args:
            holdout_labs: List of specific labs to hold out, or None for automatic selection
            n_holdout_labs: Number of labs to hold out if not specified

        Returns:
            train_df, val_df
        """
        print("\n=== Lab Hold-Out Split ===")

        if holdout_labs is None:
            # Select labs with medium representation (not too small, not too large)
            lab_counts_sorted = self.lab_counts.sort_values()
            # Select from middle of distribution
            mid_start = len(lab_counts_sorted) // 3
            mid_end = 2 * len(lab_counts_sorted) // 3
            candidate_labs = lab_counts_sorted.iloc[mid_start:mid_end].index.tolist()

            # Randomly select labs
            np.random.shuffle(candidate_labs)
            holdout_labs = candidate_labs[:n_holdout_labs]

        print(f"Holding out labs: {holdout_labs}")

        val_df = self.df[self.df['lab_id'].isin(holdout_labs)].reset_index(drop=True)
        train_df = self.df[~self.df['lab_id'].isin(holdout_labs)].reset_index(drop=True)

        print(f"Train: {len(train_df)} videos from {train_df['lab_id'].nunique()} labs")
        print(f"Val: {len(val_df)} videos from {val_df['lab_id'].nunique()} labs")

        return train_df, val_df

    def group_kfold_split(self, n_folds=5):
        """
        Create k-fold cross-validation splits with labs as groups
        Ensures videos from same lab don't appear in both train and val

        Args:
            n_folds: Number of folds

        Returns:
            List of (train_df, val_df) tuples for each fold
        """
        print(f"\n=== Group K-Fold Split ({n_folds} folds) ===")

        gkf = GroupKFold(n_splits=n_folds)
        X = self.df.index.values
        y = self.df['lab_id'].values  # Not used for splitting, just placeholder
        groups = self.df['lab_id'].values

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            val_df = self.df.iloc[val_idx].reset_index(drop=True)

            print(f"  Fold {fold_idx + 1}: Train {len(train_df)} videos ({train_df['lab_id'].nunique()} labs), "
                  f"Val {len(val_df)} videos ({val_df['lab_id'].nunique()} labs)")

            folds.append((train_df, val_df))

        return folds

    def temporal_aware_split(self, val_ratio=0.15):
        """
        Split with awareness of temporal nature - uses last portion of each video for validation
        This is applied after the main split strategy

        Args:
            val_ratio: Proportion of each video to use for validation

        Returns:
            Dictionary with split information for temporal segmentation
        """
        print("\n=== Temporal-Aware Split Info ===")
        print(f"Using last {val_ratio*100:.0f}% of each video for validation")

        temporal_splits = {}

        # For each video, define train/val frame ranges
        for idx, row in self.df.iterrows():
            video_id = row['video_id']
            duration = row['video_duration_sec']
            fps = row['frames_per_second']
            total_frames = int(duration * fps)

            val_frames = int(total_frames * val_ratio)
            train_frames = total_frames - val_frames

            temporal_splits[video_id] = {
                'train_range': (0, train_frames),
                'val_range': (train_frames, total_frames),
                'total_frames': total_frames
            }

        return temporal_splits

    def save_split(self, train_df, val_df, output_dir='datamount/splits', split_name='default'):
        """
        Save train/val splits to CSV files

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            output_dir: Directory to save splits
            split_name: Name for this split configuration
        """
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, f'{split_name}_train.csv')
        val_path = os.path.join(output_dir, f'{split_name}_val.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        # Save split metadata
        metadata = {
            'split_name': split_name,
            'n_train': len(train_df),
            'n_val': len(val_df),
            'train_labs': train_df['lab_id'].nunique(),
            'val_labs': val_df['lab_id'].nunique(),
            'seed': self.seed
        }

        metadata_path = os.path.join(output_dir, f'{split_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSplit saved to {output_dir}/{split_name}_*.csv")
        print(f"Metadata saved to {metadata_path}")

    def create_recommended_split(self):
        """
        Create the recommended split configuration for MABe challenge
        Combines lab-stratified split with temporal awareness
        """
        print("\n" + "="*50)
        print("CREATING RECOMMENDED SPLIT FOR MABe CHALLENGE")
        print("="*50)

        # 1. Lab-stratified split for main train/val division
        train_df, val_df = self.lab_stratified_split(val_ratio=0.15)

        # 2. Get temporal split info (optional - for within-video temporal validation)
        temporal_splits = self.temporal_aware_split(val_ratio=0.15)

        # 3. Save the splits
        self.save_split(train_df, val_df, split_name='lab_stratified')

        # 4. Also save temporal split info
        temporal_path = 'datamount/splits/temporal_splits.json'
        with open(temporal_path, 'w') as f:
            json.dump(temporal_splits, f, indent=2)

        print(f"\nTemporal split info saved to {temporal_path}")

        return train_df, val_df, temporal_splits

    def lab_specific_crossfold_split(self, n_folds=4, output_dir='datamount/splits/labs_crossfold'):
        """
        Create N-fold cross-validation splits for each lab individually.
        Saves folds into subdirectories named after each lab.

        Args:
            n_folds (int): Number of folds to create for each lab.
            output_dir (str): The base directory to save the lab-specific fold folders.
        """
        print(f"\n=== Lab-Specific N-Fold Split ({n_folds} folds) ===")
        os.makedirs(output_dir, exist_ok=True)

        # We use KFold because we are splitting the data within a single lab
        # We shuffle to ensure random distribution of videos in folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        for lab in self.labs:
            lab_dir = os.path.join(output_dir, str(lab))
            os.makedirs(lab_dir, exist_ok=True)

            # Get all data for the current lab
            lab_df = self.df[self.df['lab_id'] == lab].reset_index(drop=True)

            if len(lab_df) < n_folds:
                print(f"  Skipping {lab}: Only {len(lab_df)} videos, but {n_folds} folds requested.")
                continue

            print(f"  Processing {lab}: {len(lab_df)} videos")

            # kf.split() returns indices based on the input dataframe (lab_df)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(lab_df)):
                train_fold_df = lab_df.iloc[train_idx]
                val_fold_df = lab_df.iloc[val_idx]

                # Define file paths
                train_path = os.path.join(lab_dir, f'train_{fold_idx}.csv')
                val_path = os.path.join(lab_dir, f'val_{fold_idx}.csv')

                # Save the dataframes
                train_fold_df.to_csv(train_path, index=False)
                val_fold_df.to_csv(val_path, index=False)

            print(f"    Saved {n_folds} folds to {lab_dir}")

        print(f"\nLab-specific cross-validation splits saved to {output_dir}")

    def lab_stratified_crossfold_split(self, n_folds=5):
        print(f"\n=== Lab-Stratified N-Fold Split ({n_folds} folds) ===")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        X = self.df.index.values
        y = self.df['lab_id'].values

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            val_df = self.df.iloc[val_idx].reset_index(drop=True)

            print(f"  Fold {fold_idx + 1}: Train {len(train_df)} videos ({train_df['lab_id'].nunique()} labs), "
                  f"Val {len(val_df)} videos ({val_df['lab_id'].nunique()} labs)")

            folds.append((train_df, val_df))

        # save folds in splits directory
        output_dir = 'datamount/splits/lab_stratified_crossfold'
        os.makedirs(output_dir, exist_ok=True)
        for fold_idx, (train_df, val_df) in enumerate(folds):
            train_path = os.path.join(output_dir, f'train_fold{fold_idx}.csv')
            val_path = os.path.join(output_dir, f'val_fold{fold_idx}.csv')
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            print(f"    Saved Fold {fold_idx + 1} to {output_dir}")

def main():
    """Main function to demonstrate different splitting strategies"""

    splitter = MABeDataSplitter(seed=1994)

    # 1. Create recommended split
    print("\n" + "="*60)
    print("1. RECOMMENDED SPLIT (Lab-Stratified)")
    print("="*60)
    train_df, val_df, temporal_splits = splitter.create_recommended_split()

    # # 2. Alternative: Lab hold-out
    # print("\n" + "="*60)
    # print("2. ALTERNATIVE: Lab Hold-Out Split")
    # print("="*60)
    # train_df_holdout, val_df_holdout = splitter.lab_holdout_split(n_holdout_labs=3)
    # splitter.save_split(train_df_holdout, val_df_holdout, split_name='lab_holdout')

    # # 3. For cross-validation experiments
    # print("\n" + "="*60)
    # print("3. CROSS-VALIDATION: Group K-Fold")
    # print("="*60)
    # folds = splitter.group_kfold_split(n_folds=5)

    # # Save first fold as example
    # if folds:
    #     splitter.save_split(folds[0][0], folds[0][1], split_name='gkfold_fold1')

    # 4. NEW: Lab-specific n-fold cross-validation
    print("\n" + "="*60)
    print("4. LAB-SPECIFIC N-FOLD CROSS-VALIDATION")
    print("="*60)
    splitter.lab_specific_crossfold_split(n_folds=4) # Using 4 folds as requested

    # 5. NEW: Lab-stratified n-fold cross-validation
    splitter.lab_stratified_crossfold_split(n_folds=4) # Using 4 folds as requested

    print("\n" + "="*60)
    print("SPLITTING COMPLETE!")
    print("="*60)
    print("\nRecommendations:")
    print("1. Use 'lab_stratified' split for main model development")
    print("2. Use 'lab_holdout' to test generalization to new labs")
    print("3. Use group k-fold for robust performance estimation")
    print("4. Use 'labs_crossfold' splits for per-lab performance tuning")
    print("\nFiles saved in 'datamount/splits/' directory")


if __name__ == "__main__":
    main()