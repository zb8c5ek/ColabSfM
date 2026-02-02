#!/usr/bin/env python3
"""
ESSN Merge Reconstruction Script

Standalone script to register and merge two COLMAP reconstructions using ColabSfM.

Usage:
    python ESSN_merge_reconstruction.py <reconstruction_A_path> <reconstruction_B_path> <output_path> [options]

Example:
    python ESSN_merge_reconstruction.py path/to/rec_A path/to/rec_B path/to/output --mode se3
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import torch
import pycolmap

# Add parent directory to path to import colabsfm
sys.path.insert(0, str(Path(__file__).parent.parent))

from colabsfm.api import RefineRoITr
from colabsfm.utils import extract_pcd_from_colmap_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_reconstructions(reconstruction_A: pycolmap.Reconstruction,
                          reconstruction_B: pycolmap.Reconstruction,
                          transformation: np.ndarray) -> pycolmap.Reconstruction:
    """
    Merge two COLMAP reconstructions after aligning A to B using the transformation.
    
    Args:
        reconstruction_A: First reconstruction (will be transformed)
        reconstruction_B: Second reconstruction (target frame)
        transformation: 4x4 transformation matrix from A to B's coordinate frame
        
    Returns:
        Merged reconstruction in B's coordinate frame
    """
    logger.info("Transforming reconstruction A to align with B...")
    
    # Create a copy of reconstruction A and transform it
    aligned_A = pycolmap.Reconstruction(reconstruction_A)
    
    # Use pycolmap's built-in transform method
    from pycolmap import Rigid3d
    aligned_A.transform(Rigid3d(transformation))
    
    logger.info(f"Aligned reconstruction A: {len(aligned_A.cameras)} cameras, "
                f"{len(aligned_A.images)} images, {len(aligned_A.points3D)} points")
    
    # Create merged reconstruction starting with B
    logger.info("Merging reconstructions...")
    merged = pycolmap.Reconstruction(reconstruction_B)
    
    # Prepare ID mappings for handling conflicts
    max_camera_id = max(merged.cameras.keys()) if len(merged.cameras) > 0 else 0
    max_image_id = max(merged.images.keys()) if len(merged.images) > 0 else 0
    max_point3D_id = max(merged.points3D.keys()) if len(merged.points3D) > 0 else 0
    
    camera_id_map = {}
    image_id_map = {}
    point3D_id_map = {}
    
    # Merge cameras from A
    logger.info("Merging cameras...")
    for cam_id, camera in aligned_A.cameras.items():
        if cam_id in merged.cameras:
            # Assign new ID to avoid conflict
            max_camera_id += 1
            new_id = max_camera_id
            camera_id_map[cam_id] = new_id
        else:
            camera_id_map[cam_id] = cam_id
        
        # Add camera with potentially new ID
        merged.add_camera(camera)
        if cam_id != camera_id_map[cam_id]:
            # Move to new ID
            merged.cameras[camera_id_map[cam_id]] = merged.cameras.pop(cam_id)
    
    logger.info(f"Added {len(aligned_A.cameras)} cameras from A")
    
    # Merge images from A
    logger.info("Merging images...")
    for img_id, image in aligned_A.images.items():
        if img_id in merged.images:
            # Assign new ID
            max_image_id += 1
            new_id = max_image_id
            image_id_map[img_id] = new_id
        else:
            image_id_map[img_id] = img_id
        
        # Create new image with updated camera ID
        new_image = pycolmap.Image(
            id=image_id_map[img_id],
            name=image.name,
            camera_id=camera_id_map[image.camera_id],
            cam_from_world=image.cam_from_world
        )
        new_image.points2D = image.points2D
        
        merged.add_image(new_image)
    
    logger.info(f"Added {len(aligned_A.images)} images from A")
    
    # Merge 3D points from A
    logger.info("Merging 3D points...")
    for pt_id, point3D in aligned_A.points3D.items():
        if pt_id in merged.points3D:
            # Assign new ID
            max_point3D_id += 1
            new_id = max_point3D_id
            point3D_id_map[pt_id] = new_id
        else:
            point3D_id_map[pt_id] = pt_id
        
        # Add the 3D point
        merged.add_point3D(
            xyz=point3D.xyz,
            track=pycolmap.Track(),
            color=point3D.color
        )
        
        # Add track observations with remapped image IDs
        for element in point3D.track.elements:
            if element.image_id in image_id_map:
                try:
                    merged.add_observation(
                        point3D_id_map[pt_id],
                        image_id_map[element.image_id],
                        element.point2D_idx
                    )
                except:
                    # Skip if observation cannot be added
                    pass
    
    logger.info(f"Added {len(aligned_A.points3D)} 3D points from A")
    logger.info(f"Final merged reconstruction: {len(merged.cameras)} cameras, "
                f"{len(merged.images)} images, {len(merged.points3D)} 3D points")
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge two COLMAP reconstructions using ColabSfM registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two reconstructions using SE(3) alignment
  python ESSN_merge_reconstruction.py rec_A/ rec_B/ output/ --mode se3
  
  # Merge using Sim(3) alignment (allows scale changes)
  python ESSN_merge_reconstruction.py rec_A/ rec_B/ output/ --mode sim3
  
  # Use custom weights
  python ESSN_merge_reconstruction.py rec_A/ rec_B/ output/ --weights weights.pth
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
        default='se3',
        help='Registration mode: se3 (rigid) or sim3 (with scale). Default: se3'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights. If not specified, downloads default weights.'
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
    
    # Initialize registrator
    logger.info(f"Initializing ColabSfM registrator (mode: {args.mode})...")
    try:
        registrator = RefineRoITr(
            weights_path=args.weights,
            mode=args.mode
        )
    except Exception as e:
        logger.error(f"Failed to initialize registrator: {e}")
        return 1
    
    # Register reconstructions - this returns only the transformation, not a merged model
    logger.info("Computing alignment transformation...")
    try:
        results = registrator.register_reconstructions(reconstruction_A, reconstruction_B)
        transformation = results['transformation']
        num_matches = results['num_matches']
        
        logger.info(f"Registration successful!")
        logger.info(f"  Number of matches: {num_matches}")
        logger.info(f"  Transformation matrix:")
        for row in transformation:
            logger.info(f"    {row}")
    except Exception as e:
        logger.error(f"Failed to register reconstructions: {e}")
        return 1
    
    # Merge reconstructions
    logger.info("Merging reconstructions...")
    try:
        merged_reconstruction = merge_reconstructions(
            reconstruction_A,
            reconstruction_B,
            transformation
        )
    except Exception as e:
        logger.error(f"Failed to merge reconstructions: {e}")
        return 1
    
    # Save merged reconstruction
    logger.info(f"Saving merged reconstruction to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    
    try:
        if args.format == 'bin':
            merged_reconstruction.write_binary(args.output)
        else:
            merged_reconstruction.write_text(args.output)
        logger.info(f"Successfully saved merged reconstruction ({args.format} format)")
    except Exception as e:
        logger.error(f"Failed to save merged reconstruction: {e}")
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
