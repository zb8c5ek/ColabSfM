#!/usr/bin/env python3
"""
Cascade Merge Script for sample_1 dataset

Merges three bubble reconstructions in a cascade:
  1. bubble_00 + bubble_01 → merged_00_01
  2. bubble_00 + bubble_02 → merged_00_02  
  3. merged_00_01 + merged_00_02 → final_merged

Uses bubble_00 as the common reference frame for alignment.
"""

import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pycolmap

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from ESSN_merge_reconstruction import (
    find_matching_images,
    extract_camera_correspondences,
    merge_reconstructions
)
from kern_umeyama_solver import ransac_umeyama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_merge(rec_A_path: str, rec_B_path: str, output_path: str, 
              mode: str = 'sim3', threshold: float = 0.1, max_iter: int = 10000) -> Dict[str, Any]:
    """
    Run a single merge operation and return debug info.
    
    Args:
        rec_A_path: Path to reconstruction A (source, will be transformed)
        rec_B_path: Path to reconstruction B (target, reference frame)
        output_path: Output directory for merged result
        mode: 'sim3' or 'se3'
        threshold: RANSAC inlier threshold
        max_iter: RANSAC max iterations
    
    Returns:
        Dict with debug info and path to merged reconstruction
    """
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'rec_A_path': rec_A_path,
        'rec_B_path': rec_B_path,
        'output_path': output_path,
        'mode': mode,
        'ransac_threshold': threshold,
        'ransac_max_iterations': max_iter,
    }
    
    logger.info(f"Merging: {os.path.basename(rec_A_path)} -> {os.path.basename(rec_B_path)}")
    
    # Load reconstructions
    rec_A = pycolmap.Reconstruction(rec_A_path)
    rec_B = pycolmap.Reconstruction(rec_B_path)
    
    debug_info['rec_A'] = {
        'cameras': len(rec_A.cameras),
        'images': len(rec_A.images),
        'points3D': len(rec_A.points3D)
    }
    debug_info['rec_B'] = {
        'cameras': len(rec_B.cameras),
        'images': len(rec_B.images),
        'points3D': len(rec_B.points3D)
    }
    
    logger.info(f"  Rec A: {len(rec_A.cameras)} cams, {len(rec_A.images)} imgs, {len(rec_A.points3D)} pts")
    logger.info(f"  Rec B: {len(rec_B.cameras)} cams, {len(rec_B.images)} imgs, {len(rec_B.points3D)} pts")
    
    # Find matching images
    matching_images = find_matching_images(rec_A, rec_B)
    debug_info['matching_images'] = {
        'count': len(matching_images),
        'names': [name for _, _, name in matching_images]
    }
    logger.info(f"  Found {len(matching_images)} matching images")
    
    if len(matching_images) < 3:
        raise ValueError(f"Not enough matching images ({len(matching_images)} < 3)")
    
    # Extract camera correspondences
    centers_A, centers_B = extract_camera_correspondences(rec_A, rec_B, matching_images)
    
    # Compute transformation with RANSAC - get detailed results
    R, t, s = ransac_umeyama(
        centers_A, centers_B,
        npoints=3,
        threshold=threshold,
        max_iter=max_iter,
        compute_rmse=False
    )
    
    # Force scale=1 for SE3 mode
    if mode == 'se3':
        s = 1.0
    
    # Compute inliers and errors
    points_A_transformed = s * (centers_A @ R.T) + t
    errors = np.linalg.norm(points_A_transformed - centers_B, axis=1)
    inlier_mask = errors < threshold
    inlier_count = int(inlier_mask.sum())
    inlier_rmse = float(np.sqrt(np.mean(errors[inlier_mask]**2))) if inlier_count > 0 else 0.0
    
    # Build 4x4 transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = s * R
    transformation[:3, 3] = t
    
    # Store RANSAC results
    debug_info['ransac_result'] = {
        'scale': float(s),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'transformation_4x4': transformation.tolist(),
        'inliers': {
            'count': inlier_count,
            'total': len(centers_A),
            'ratio': float(inlier_count / len(centers_A)),
            'indices': [int(i) for i in np.where(inlier_mask)[0]],
            'image_names': [matching_images[i][2] for i in np.where(inlier_mask)[0]]
        },
        'inlier_rmse': inlier_rmse,
        'all_errors': errors.tolist()
    }
    
    logger.info(f"  Scale: {s:.6f}, Inliers: {inlier_count}/{len(centers_A)} ({inlier_count/len(centers_A)*100:.1f}%), RMSE: {inlier_rmse:.6f}")
    
    # Merge
    os.makedirs(output_path, exist_ok=True)
    merged_path = os.path.join(output_path, 'merged')
    merge_reconstructions(rec_A, rec_B, transformation, merged_path)
    
    # Load merged result for stats
    merged_rec = pycolmap.Reconstruction(merged_path)
    debug_info['merged'] = {
        'cameras': len(merged_rec.cameras),
        'images': len(merged_rec.images),
        'points3D': len(merged_rec.points3D),
        'path': merged_path
    }
    
    # Save debug info to JSON
    debug_json_path = os.path.join(output_path, 'debug_info.json')
    with open(debug_json_path, 'w') as f:
        json.dump(debug_info, f, indent=2)
    logger.info(f"  Debug info saved to: {debug_json_path}")
    
    return debug_info


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    BASE_DIR = r"D:\ColabSfM\MOD_MergeReconstruction\_data\sample_1"
    
    bubble_00 = os.path.join(BASE_DIR, "bubble_00", "sparse", "0")
    bubble_01 = os.path.join(BASE_DIR, "bubble_01", "sparse", "0")
    bubble_02 = os.path.join(BASE_DIR, "bubble_02", "sparse", "0")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, f"cascade_merged_{timestamp}")
    
    # Merge parameters
    MODE = 'sim3'
    THRESHOLD = 0.1
    MAX_ITER = 10000
    
    # =========================================================================
    logger.info("=" * 70)
    logger.info("CASCADE MERGE: sample_1 dataset")
    logger.info("=" * 70)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")
    
    # =========================================================================
    # Stage 1: Merge bubble_00 + bubble_01
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 1: Merge bubble_00 + bubble_01")
    logger.info("=" * 70)
    
    stage1_info = run_merge(
        rec_A_path=bubble_01,  # Source (will be transformed)
        rec_B_path=bubble_00,  # Target (reference frame)
        output_path=os.path.join(OUTPUT_DIR, "stage1_00_01"),
        mode=MODE,
        threshold=THRESHOLD,
        max_iter=MAX_ITER
    )
    merged_00_01_path = stage1_info['merged']['path']
    logger.info(f"Stage 1 complete: {merged_00_01_path}")
    logger.info("")
    
    # =========================================================================
    # Stage 2: Merge bubble_00 + bubble_02
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 2: Merge bubble_00 + bubble_02")
    logger.info("=" * 70)
    
    stage2_info = run_merge(
        rec_A_path=bubble_02,  # Source (will be transformed)
        rec_B_path=bubble_00,  # Target (reference frame)
        output_path=os.path.join(OUTPUT_DIR, "stage2_00_02"),
        mode=MODE,
        threshold=THRESHOLD,
        max_iter=MAX_ITER
    )
    merged_00_02_path = stage2_info['merged']['path']
    logger.info(f"Stage 2 complete: {merged_00_02_path}")
    logger.info("")
    
    # =========================================================================
    # Stage 3: Merge the two merged reconstructions
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STAGE 3: Merge stage1 + stage2 -> final")
    logger.info("=" * 70)
    
    # merged_00_01 has bubble_00 as reference, so use it as target
    # merged_00_02 needs to be transformed to match
    stage3_info = run_merge(
        rec_A_path=merged_00_02_path,  # Source
        rec_B_path=merged_00_01_path,  # Target (has bubble_00 reference)
        output_path=os.path.join(OUTPUT_DIR, "final"),
        mode=MODE,
        threshold=THRESHOLD,
        max_iter=MAX_ITER
    )
    final_merged_path = stage3_info['merged']['path']
    logger.info(f"Stage 3 complete: {final_merged_path}")
    logger.info("")
    
    # =========================================================================
    # Save cascade summary
    # =========================================================================
    cascade_summary = {
        'timestamp': datetime.now().isoformat(),
        'output_dir': OUTPUT_DIR,
        'config': {
            'mode': MODE,
            'ransac_threshold': THRESHOLD,
            'ransac_max_iterations': MAX_ITER
        },
        'inputs': {
            'bubble_00': bubble_00,
            'bubble_01': bubble_01,
            'bubble_02': bubble_02
        },
        'stages': {
            'stage1_00_01': {
                'scale': stage1_info['ransac_result']['scale'],
                'inlier_ratio': stage1_info['ransac_result']['inliers']['ratio'],
                'inlier_rmse': stage1_info['ransac_result']['inlier_rmse'],
                'merged_images': stage1_info['merged']['images'],
                'merged_points': stage1_info['merged']['points3D']
            },
            'stage2_00_02': {
                'scale': stage2_info['ransac_result']['scale'],
                'inlier_ratio': stage2_info['ransac_result']['inliers']['ratio'],
                'inlier_rmse': stage2_info['ransac_result']['inlier_rmse'],
                'merged_images': stage2_info['merged']['images'],
                'merged_points': stage2_info['merged']['points3D']
            },
            'stage3_final': {
                'scale': stage3_info['ransac_result']['scale'],
                'inlier_ratio': stage3_info['ransac_result']['inliers']['ratio'],
                'inlier_rmse': stage3_info['ransac_result']['inlier_rmse'],
                'merged_images': stage3_info['merged']['images'],
                'merged_points': stage3_info['merged']['points3D']
            }
        },
        'final_result': stage3_info['merged']
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'cascade_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(cascade_summary, f, indent=2)
    logger.info(f"Cascade summary saved to: {summary_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 70)
    logger.info("CASCADE MERGE COMPLETE")
    logger.info("=" * 70)
    
    # Load and display final stats
    final_rec = pycolmap.Reconstruction(final_merged_path)
    logger.info(f"Final merged reconstruction:")
    logger.info(f"  Cameras: {len(final_rec.cameras)}")
    logger.info(f"  Images:  {len(final_rec.images)}")
    logger.info(f"  Points:  {len(final_rec.points3D)}")
    logger.info(f"  Location: {final_merged_path}")
    logger.info(f"  Summary JSON: {summary_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
