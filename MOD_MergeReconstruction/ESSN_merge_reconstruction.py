#!/usr/bin/env python3
"""
ESSN Merge Reconstruction Script

Standalone script to register and merge two COLMAP reconstructions using Umeyama SIM3 solver.
Uses camera poses from images with matching names as correspondences for alignment.

Usage:
    python ESSN_merge_reconstruction.py <reconstruction_A_path> <reconstruction_B_path> <output_path> [options]

Example:
    python ESSN_merge_reconstruction.py path/to/rec_A path/to/rec_B path/to/output --mode sim3

pycolmap API Reference:
    rec = pycolmap.Reconstruction(path)     # Load sparse model (binary or text)
    rec.images -> Dict[int, Image]          # image_id -> Image mapping
    rec.cameras -> Dict[int, Camera]        # camera_id -> Camera mapping
    rec.points3D -> Dict[int, Point3D]      # point3D_id -> Point3D mapping

    image.name -> str                       # Image filename
    image.has_pose -> bool                  # True if pose is registered (attribute, not method)
    image.cam_from_world() -> Rigid3d       # Camera pose (METHOD - call with parentheses)
    image.projection_center() -> np.ndarray # Camera center in world coords (3,)

    pycolmap.Sim3d(matrix_3x4)              # Construct from [sR | t] matrix
    rec.transform(Sim3d)                    # Transform reconstruction in-place
    rec.write_binary(path)                  # Save as binary (cameras.bin, images.bin, points3D.bin, ...)
    rec.write_text(path)                    # Save as text format
"""

import argparse
import copy
import logging
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pycolmap

# Import local utilities
from kern_umeyama_solver import ransac_umeyama, umeyama
from kern_colmap_utils import (
    write_reconstruction_txt,
    pycolmap_image_to_dict,
    pycolmap_point3d_to_dict
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_matching_images(reconstruction_A: pycolmap.Reconstruction,
                         reconstruction_B: pycolmap.Reconstruction) -> List[Tuple[int, int, str]]:
    """
    Find images with matching names between two reconstructions.
    
    Args:
        reconstruction_A: First reconstruction
        reconstruction_B: Second reconstruction
    
    Returns:
        List of tuples (image_id_A, image_id_B, image_name)
    """
    # Build name to image_id mapping for reconstruction B
    name_to_id_B = {img.name: img_id for img_id, img in reconstruction_B.images.items()}
    
    matching_images = []
    for img_id_A, img_A in reconstruction_A.images.items():
        if img_A.name in name_to_id_B:
            img_id_B = name_to_id_B[img_A.name]
            matching_images.append((img_id_A, img_id_B, img_A.name))
    
    return matching_images


def load_point_correspondences(matches_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed 3D point correspondences from file.
    
    NOTE: This function is reserved for future use when 3D point matching is supported.
    
    Args:
        matches_file: Path to numpy file containing matched points.
                      Should be .npz file with keys 'points_A' and 'points_B',
                      or .npy file with shape (N, 6) where [:, :3] is points_A and [:, 3:] is points_B
    
    Returns:
        Tuple of (points_A, points_B), each with shape (N, 3)
    """
    if matches_file.endswith('.npz'):
        data = np.load(matches_file)
        points_A = data['points_A']
        points_B = data['points_B']
    elif matches_file.endswith('.npy'):
        data = np.load(matches_file)
        if data.shape[1] == 6:
            points_A = data[:, :3]
            points_B = data[:, 3:]
        else:
            raise ValueError(f"Expected .npy file with shape (N, 6), got {data.shape}")
    else:
        raise ValueError(f"Unsupported file format: {matches_file}. Use .npz or .npy")
    
    if len(points_A) == 0:
        raise ValueError("No point correspondences found in file")
    
    logger.info(f"Loaded {len(points_A)} point correspondences from {matches_file}")
    return points_A, points_B


def get_image_camera_center(image: pycolmap.Image) -> np.ndarray:
    """
    Get the camera center (position in world coordinates) for an image.
    
    Args:
        image: pycolmap Image object
    
    Returns:
        Camera center as numpy array of shape (3,)
    """
    # Use pycolmap's built-in projection_center method
    return image.projection_center()


def extract_camera_correspondences(
    reconstruction_A: pycolmap.Reconstruction,
    reconstruction_B: pycolmap.Reconstruction,
    matching_images: List[Tuple[int, int, str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera center correspondences from matching images.
    
    Args:
        reconstruction_A: First reconstruction
        reconstruction_B: Second reconstruction
        matching_images: List of (image_id_A, image_id_B, image_name) tuples
    
    Returns:
        Tuple of (centers_A, centers_B), each with shape (N, 3)
    """
    centers_A = []
    centers_B = []
    
    for img_id_A, img_id_B, img_name in matching_images:
        img_A = reconstruction_A.images[img_id_A]
        img_B = reconstruction_B.images[img_id_B]
        
        center_A = get_image_camera_center(img_A)
        center_B = get_image_camera_center(img_B)
        
        centers_A.append(center_A)
        centers_B.append(center_B)
    
    return np.array(centers_A), np.array(centers_B)


def compute_transformation(points_A: np.ndarray,
                          points_B: np.ndarray,
                          mode: str = 'sim3',
                          ransac_threshold: float = 0.1,
                          ransac_iterations: int = 10000) -> np.ndarray:
    """
    Compute transformation from A to B using Umeyama algorithm.
    
    Args:
        points_A: Source points from reconstruction A, shape (N, 3)
        points_B: Target points from reconstruction B, shape (N, 3)
        mode: 'se3' (rigid) or 'sim3' (with scale)
        ransac_threshold: RANSAC inlier threshold
        ransac_iterations: Maximum RANSAC iterations
    
    Returns:
        4x4 transformation matrix from A to B coordinate frame
    """
    logger.info(f"Computing {mode.upper()} transformation using Umeyama algorithm...")
    logger.info(f"  Number of point correspondences: {len(points_A)}")
    logger.info(f"  RANSAC threshold: {ransac_threshold}")
    
    # Compute transformation using RANSAC + Umeyama
    R, t, s = ransac_umeyama(
        points_A,
        points_B,
        npoints=3,
        threshold=ransac_threshold,
        max_iter=ransac_iterations,
        compute_rmse=False
    )
    
    # For SE3 mode, force scale to 1
    if mode == 'se3':
        s = 1.0
        logger.info(f"  SE3 mode: forcing scale to 1.0")
    else:
        logger.info(f"  Estimated scale: {s:.6f}")
    
    # Build 4x4 transformation matrix
    # Transformation: p_B = s * R * p_A + t
    transformation = np.eye(4)
    transformation[:3, :3] = s * R
    transformation[:3, 3] = t
    
    # Compute final error
    points_A_transformed = s * (points_A @ R.T) + t
    errors = np.linalg.norm(points_A_transformed - points_B, axis=1)
    inliers = errors < ransac_threshold
    
    logger.info(f"  Rotation matrix:")
    for row in R:
        logger.info(f"    {row}")
    logger.info(f"  Translation: {t}")
    logger.info(f"  Number of inliers: {inliers.sum()} / {len(points_A)} ({inliers.sum()/len(points_A)*100:.1f}%)")
    logger.info(f"  Inlier RMSE: {np.sqrt(np.mean(errors[inliers]**2)):.6f}")
    
    return transformation


def merge_reconstructions(reconstruction_A: pycolmap.Reconstruction,
                          reconstruction_B: pycolmap.Reconstruction,
                          transformation: np.ndarray,
                          output_path: str) -> Tuple[pycolmap.Reconstruction, pycolmap.Reconstruction]:
    """
    Merge two COLMAP reconstructions after aligning A to B using the transformation.
    Writes the merged reconstruction in COLMAP text format to handle pycolmap limitations.
    
    Args:
        reconstruction_A: First reconstruction (will be transformed)
        reconstruction_B: Second reconstruction (target frame)
        transformation: 4x4 transformation matrix from A to B's coordinate frame
                       (includes scale in the rotation part: sR)
        output_path: Path to write the merged reconstruction
        
    Returns:
        Tuple of (merged_points_only, aligned_A) reconstructions
    """
    logger.info("Transforming reconstruction A to align with B...")
    
    # Create a copy of reconstruction A and transform it
    aligned_A = pycolmap.Reconstruction(reconstruction_A)
    
    # Use pycolmap's Sim3d for transformation (supports scale)
    sim3_matrix = transformation[:3, :]
    sim3_transform = pycolmap.Sim3d(sim3_matrix)
    
    logger.info(f"  Sim3d transform: scale={sim3_transform.scale:.6f}")
    aligned_A.transform(sim3_transform)
    
    logger.info(f"Aligned reconstruction A: {len(aligned_A.cameras)} cameras, "
                f"{len(aligned_A.images)} images, {len(aligned_A.points3D)} points")
    
    # Merge and write reconstruction using kern_colmap_utils
    logger.info("Writing merged reconstruction in COLMAP text format...")
    os.makedirs(output_path, exist_ok=True)
    
    # Build set of existing image names in B to detect duplicates
    existing_image_names_set = {img.name for img in reconstruction_B.images.values()}
    
    # ===== Merge cameras (0-based IDs) =====
    camera_id_map_B = {}  # old_id_B -> new_id
    camera_id_map_A = {}  # old_id_A -> new_id
    all_cameras = []
    new_cam_id = 0
    
    for old_cam_id, camera in reconstruction_B.cameras.items():
        camera_id_map_B[old_cam_id] = new_cam_id
        all_cameras.append((new_cam_id, camera))
        new_cam_id += 1
    
    for old_cam_id, camera in aligned_A.cameras.items():
        camera_id_map_A[old_cam_id] = new_cam_id
        all_cameras.append((new_cam_id, camera))
        new_cam_id += 1
    
    # ===== Merge images (0-based IDs, skip duplicates) =====
    all_images = []
    new_img_id = 0
    images_added = 0
    images_skipped = 0
    
    for img_id, image in reconstruction_B.images.items():
        all_images.append(pycolmap_image_to_dict(image, new_img_id, camera_id_map_B[image.camera_id]))
        new_img_id += 1
    
    for img_id, image in aligned_A.images.items():
        if image.name in existing_image_names_set:
            images_skipped += 1
            continue
        all_images.append(pycolmap_image_to_dict(image, new_img_id, camera_id_map_A[image.camera_id]))
        new_img_id += 1
        images_added += 1
    
    # ===== Merge 3D points (0-based IDs, no tracks) =====
    all_points = []
    new_pt_id = 0
    
    for pt_id, point in reconstruction_B.points3D.items():
        all_points.append({'point3D_id': new_pt_id, 'xyz': point.xyz, 'color': point.color})
        new_pt_id += 1
    
    for pt_id, point in aligned_A.points3D.items():
        all_points.append({'point3D_id': new_pt_id, 'xyz': point.xyz, 'color': point.color})
        new_pt_id += 1
    
    # Write using kern_colmap_utils
    write_reconstruction_txt(all_cameras, all_images, all_points, output_path)
    
    logger.info(f"  Written {len(all_cameras)} cameras")
    logger.info(f"  Written {len(all_images)} images ({images_added} from A, {images_skipped} duplicates skipped)")
    logger.info(f"  Written {len(all_points)} 3D points")
    
    # Summary
    logger.info(f"Merged reconstruction written to: {output_path}")
    logger.info(f"  Total: {len(all_cameras)} cameras, {len(all_images)} images, {len(all_points)} 3D points")
    
    # Also create a point-only merged reconstruction for API return
    merged_points = pycolmap.Reconstruction(reconstruction_B)
    for pt_id, point in aligned_A.points3D.items():
        merged_points.add_point3D(xyz=point.xyz, track=pycolmap.Track(), color=point.color)
    
    return merged_points, aligned_A


def main():
    parser = argparse.ArgumentParser(
        description="Merge two COLMAP reconstructions using Umeyama SIM3 solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two reconstructions using SIM3 alignment (with scale)
  python ESSN_merge_reconstruction.py rec_A/ rec_B/ output/ --mode sim3
  
  # Merge using SE3 alignment (rigid, no scale)
  python ESSN_merge_reconstruction.py rec_A/ rec_B/ output/ --mode se3
  
  # Custom RANSAC parameters
  python ESSN_merge_reconstruction.py rec_A/ rec_B/ output/ --ransac-threshold 0.05 --ransac-iterations 50000

Note:
  The script finds images with matching names between both reconstructions,
  extracts their camera poses (centers), and uses them as correspondences for alignment.
        """
    )
    
    parser.add_argument(
        'reconstruction_A',
        type=str,
        help='Path to first COLMAP reconstruction (will be aligned to B)'
    )
    
    parser.add_argument(
        'reconstruction_B',
        type=str,
        help='Path to second COLMAP reconstruction (target coordinate frame)'
    )
    
    parser.add_argument(
        'output',
        type=str,
        help='Path to save merged reconstruction'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['se3', 'sim3'],
        default='sim3',
        help='Registration mode: se3 (rigid) or sim3 (with scale). Default: sim3'
    )
    
    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=0.1,
        help='RANSAC inlier threshold. Default: 0.1'
    )
    
    parser.add_argument(
        '--ransac-iterations',
        type=int,
        default=10000,
        help='Maximum RANSAC iterations. Default: 10000'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['bin', 'txt'],
        default='bin',
        help='Output format: bin (binary) or txt (text). Default: bin'
    )
    
    parser.add_argument(
        '--save-transform',
        action='store_true',
        help='Save the transformation matrix to a .npy file'
    )
    
    args = parser.parse_args()
    
    # Load reconstructions
    logger.info(f"Loading reconstruction A from: {args.reconstruction_A}")
    try:
        reconstruction_A = pycolmap.Reconstruction(args.reconstruction_A)
        logger.info(f"  Loaded: {len(reconstruction_A.cameras)} cameras, "
                   f"{len(reconstruction_A.images)} images, "
                   f"{len(reconstruction_A.points3D)} 3D points")
    except Exception as e:
        logger.error(f"Failed to load reconstruction A: {e}")
        return 1
    
    logger.info(f"Loading reconstruction B from: {args.reconstruction_B}")
    try:
        reconstruction_B = pycolmap.Reconstruction(args.reconstruction_B)
        logger.info(f"  Loaded: {len(reconstruction_B.cameras)} cameras, "
                   f"{len(reconstruction_B.images)} images, "
                   f"{len(reconstruction_B.points3D)} 3D points")
    except Exception as e:
        logger.error(f"Failed to load reconstruction B: {e}")
        return 1
    
    # Find matching images between reconstructions
    logger.info("Finding matching images between reconstructions...")
    try:
        matching_images = find_matching_images(reconstruction_A, reconstruction_B)
        logger.info(f"  Found {len(matching_images)} matching images by name")
        
        if len(matching_images) == 0:
            logger.error("No matching images found between reconstructions!")
            logger.error("Make sure both reconstructions contain images with the same names.")
            return 1
        
        # Log some example matches
        for i, (img_id_A, img_id_B, img_name) in enumerate(matching_images[:5]):
            logger.info(f"    {img_name}: A[{img_id_A}] <-> B[{img_id_B}]")
        if len(matching_images) > 5:
            logger.info(f"    ... and {len(matching_images) - 5} more")
    except Exception as e:
        logger.error(f"Failed to find matching images: {e}")
        return 1
    
    # Extract camera center correspondences from matching images
    logger.info("Extracting camera center correspondences...")
    try:
        centers_A, centers_B = extract_camera_correspondences(
            reconstruction_A,
            reconstruction_B,
            matching_images
        )
        logger.info(f"  Extracted {len(centers_A)} camera center correspondences")
    except Exception as e:
        logger.error(f"Failed to extract camera correspondences: {e}")
        return 1
    
    # Compute transformation using Umeyama algorithm
    logger.info(f"Computing {args.mode.upper()} transformation...")
    try:
        transformation = compute_transformation(
            centers_A,
            centers_B,
            mode=args.mode,
            ransac_threshold=args.ransac_threshold,
            ransac_iterations=args.ransac_iterations
        )
        
        logger.info(f"Transformation computed successfully!")
        logger.info(f"  Transformation matrix:")
        for row in transformation:
            logger.info(f"    {row}")
    except Exception as e:
        logger.error(f"Failed to compute transformation: {e}")
        return 1
    
    # Merge reconstructions
    logger.info("Merging reconstructions...")
    os.makedirs(args.output, exist_ok=True)
    merged_dir = os.path.join(args.output, 'merged')
    
    try:
        merged_reconstruction, aligned_A = merge_reconstructions(
            reconstruction_A,
            reconstruction_B,
            transformation,
            merged_dir
        )
    except Exception as e:
        logger.error(f"Failed to merge reconstructions: {e}")
        return 1
    
    # Save aligned A separately
    aligned_A_dir = os.path.join(args.output, 'aligned_A')
    os.makedirs(aligned_A_dir, exist_ok=True)
    try:
        if args.format == 'bin':
            aligned_A.write_binary(aligned_A_dir)
        else:
            aligned_A.write_text(aligned_A_dir)
        logger.info(f"Successfully saved aligned reconstruction A to {aligned_A_dir}")
    except Exception as e:
        logger.error(f"Failed to save aligned reconstruction A: {e}")
        return 1
    
    # Optionally save transformation matrix
    if args.save_transform:
        transform_path = os.path.join(args.output, 'transformation.npy')
        np.save(transform_path, transformation)
        logger.info(f"Saved transformation matrix to: {transform_path}")
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
