#!/usr/bin/env python3
"""
Test script for ESSN_merge_reconstruction

Tests merging two COLMAP reconstructions:
  - Reconstruction A: left_rig/sparse/1
  - Reconstruction B: right_rig/sparse/3

This script demonstrates the merge workflow for aligning left and right rig
reconstructions from the RopeCap0121 dataset.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ESSN_merge_reconstruction import (
    find_matching_images,
    extract_camera_correspondences,
    compute_transformation,
    merge_reconstructions
)
import pycolmap
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # ========== Configuration ==========
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path(r"D:\ColabSfM\MOD_MergeReconstruction\_data")
    
    # Input reconstructions
    rec_A_path = base_dir / "bubble_00" / "sparse" / "0"
    rec_B_path = base_dir / "bubble_01" / "sparse" / "0"
    
    # Output directory with timestamp
    output_dir = base_dir / f"merged_bubble00_bubble01_{timestamp}"
    
    # Algorithm parameters
    mode = "sim3"           # 'sim3' (with scale) or 'se3' (rigid)
    ransac_threshold = 0.1  # RANSAC inlier threshold in meters
    ransac_iterations = 10000
    output_format = "bin"   # 'bin' or 'txt'
    save_transform = True   # Save transformation matrix
    
    # ========== Validation ==========
    logger.info("=" * 60)
    logger.info("ESSN Merge Reconstruction Test Script")
    logger.info("=" * 60)
    
    logger.info(f"Reconstruction A (source): {rec_A_path}")
    logger.info(f"Reconstruction B (target): {rec_B_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if input paths exist
    if not rec_A_path.exists():
        logger.error(f"Reconstruction A path does not exist: {rec_A_path}")
        return 1
    
    if not rec_B_path.exists():
        logger.error(f"Reconstruction B path does not exist: {rec_B_path}")
        return 1
    
    # ========== Load Reconstructions ==========
    logger.info("-" * 60)
    logger.info("Loading reconstructions...")
    
    try:
        reconstruction_A = pycolmap.Reconstruction(str(rec_A_path))
        logger.info(f"  Reconstruction A: {len(reconstruction_A.cameras)} cameras, "
                   f"{len(reconstruction_A.images)} images, "
                   f"{len(reconstruction_A.points3D)} 3D points")
    except Exception as e:
        logger.error(f"Failed to load reconstruction A: {e}")
        return 1
    
    try:
        reconstruction_B = pycolmap.Reconstruction(str(rec_B_path))
        logger.info(f"  Reconstruction B: {len(reconstruction_B.cameras)} cameras, "
                   f"{len(reconstruction_B.images)} images, "
                   f"{len(reconstruction_B.points3D)} 3D points")
    except Exception as e:
        logger.error(f"Failed to load reconstruction B: {e}")
        return 1
    
    # ========== Print Image Names for Debugging ==========
    logger.info("-" * 60)
    logger.info("Sample image names in reconstruction A:")
    for i, (img_id, img) in enumerate(reconstruction_A.images.items()):
        if i < 10:
            logger.info(f"    [{img_id}] {img.name}")
        elif i == 10:
            logger.info(f"    ... and {len(reconstruction_A.images) - 10} more")
            break
    
    logger.info("Sample image names in reconstruction B:")
    for i, (img_id, img) in enumerate(reconstruction_B.images.items()):
        if i < 10:
            logger.info(f"    [{img_id}] {img.name}")
        elif i == 10:
            logger.info(f"    ... and {len(reconstruction_B.images) - 10} more")
            break
    
    # ========== Find Matching Images ==========
    logger.info("-" * 60)
    logger.info("Finding matching images between reconstructions...")
    
    matching_images = find_matching_images(reconstruction_A, reconstruction_B)
    logger.info(f"  Found {len(matching_images)} matching images by name")
    
    if len(matching_images) == 0:
        logger.error("No matching images found between reconstructions!")
        logger.error("The reconstructions may not share any common images.")
        logger.info("This is expected if left_rig and right_rig capture different views.")
        logger.info("For this case, you may need 3D point correspondences instead.")
        return 1
    
    # Log matches
    logger.info("  Matching images:")
    for i, (img_id_A, img_id_B, img_name) in enumerate(matching_images):
        if i < 10:
            logger.info(f"    {img_name}: A[{img_id_A}] <-> B[{img_id_B}]")
        elif i == 10:
            logger.info(f"    ... and {len(matching_images) - 10} more")
            break
    
    # ========== Extract Camera Correspondences ==========
    logger.info("-" * 60)
    logger.info("Extracting camera center correspondences...")
    
    centers_A, centers_B = extract_camera_correspondences(
        reconstruction_A,
        reconstruction_B,
        matching_images
    )
    logger.info(f"  Extracted {len(centers_A)} camera center correspondences")
    
    # Print some statistics
    distances_A = np.linalg.norm(centers_A - centers_A.mean(axis=0), axis=1)
    distances_B = np.linalg.norm(centers_B - centers_B.mean(axis=0), axis=1)
    logger.info(f"  Reconstruction A centroid spread: mean={distances_A.mean():.3f}, max={distances_A.max():.3f}")
    logger.info(f"  Reconstruction B centroid spread: mean={distances_B.mean():.3f}, max={distances_B.max():.3f}")
    
    # ========== Compute Transformation ==========
    logger.info("-" * 60)
    logger.info(f"Computing {mode.upper()} transformation...")
    
    transformation = compute_transformation(
        centers_A,
        centers_B,
        mode=mode,
        ransac_threshold=ransac_threshold,
        ransac_iterations=ransac_iterations
    )
    
    logger.info("Transformation matrix (4x4):")
    for row in transformation:
        logger.info(f"  {row}")
    
    # Extract scale from transformation
    sR = transformation[:3, :3]
    scale = np.linalg.norm(sR[:, 0])  # Extract scale from first column
    logger.info(f"  Extracted scale: {scale:.6f}")
    
    # ========== Merge Reconstructions ==========
    logger.info("-" * 60)
    logger.info("Merging reconstructions...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged"
    
    merged_reconstruction, aligned_A = merge_reconstructions(
        reconstruction_A,
        reconstruction_B,
        transformation,
        str(merged_dir)
    )
    
    # ========== Save Results ==========
    logger.info("-" * 60)
    logger.info(f"Saving additional results to: {output_dir}")
    
    # Save aligned reconstruction A separately (for visualization)
    aligned_A_dir = output_dir / "aligned_A"
    aligned_A_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "bin":
        aligned_A.write_binary(str(aligned_A_dir))
    else:
        aligned_A.write_text(str(aligned_A_dir))
    logger.info(f"  Saved aligned reconstruction A to: {aligned_A_dir}")
    
    # Save transformation matrix
    if save_transform:
        transform_path = output_dir / "transformation_A_to_B.npy"
        np.save(str(transform_path), transformation)
        logger.info(f"  Saved transformation matrix to: {transform_path}")
        
        # Also save as human-readable text
        transform_txt_path = output_dir / "transformation_A_to_B.txt"
        with open(transform_txt_path, 'w') as f:
            f.write("# Transformation matrix from reconstruction A to B\n")
            f.write(f"# Mode: {mode}\n")
            f.write(f"# Source: {rec_A_path}\n")
            f.write(f"# Target: {rec_B_path}\n")
            f.write(f"# Scale: {scale:.10f}\n")
            f.write("# Matrix (4x4):\n")
            for row in transformation:
                f.write(" ".join(f"{v:.10f}" for v in row) + "\n")
        logger.info(f"  Saved transformation matrix (text) to: {transform_txt_path}")
    
    # ========== Summary ==========
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Reconstruction A: {rec_A_path}")
    logger.info(f"  Reconstruction B: {rec_B_path}")
    logger.info(f"  Matching images: {len(matching_images)}")
    logger.info(f"  Registration mode: {mode}")
    logger.info(f"  Scale factor: {scale:.6f}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
