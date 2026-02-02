#!/usr/bin/env python3
"""
Split a COLMAP reconstruction into two overlapping parts for testing merge functionality.

Usage:
    python split_reconstruction.py <input_reconstruction> <output_dir> [--overlap 0.3]
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import pycolmap


def split_reconstruction(reconstruction, overlap_ratio=0.3):
    """
    Split a reconstruction into two parts with overlap.
    
    Args:
        reconstruction: Input pycolmap.Reconstruction
        overlap_ratio: Ratio of images to overlap (0.0 to 1.0)
    
    Returns:
        rec_A, rec_B: Two new reconstructions with overlap
    """
    image_ids = sorted(reconstruction.images.keys())
    n_images = len(image_ids)
    
    if n_images < 3:
        raise ValueError(f"Need at least 3 images, got {n_images}")
    
    # Calculate split indices
    overlap_count = int(n_images * overlap_ratio)
    split_point = n_images // 2
    
    # Group A: first half + overlap into second half
    # Group B: overlap from first half + second half
    ids_A = image_ids[:split_point + overlap_count]
    ids_B = image_ids[split_point - overlap_count:]
    
    print(f"Total images: {n_images}")
    print(f"Reconstruction A: {len(ids_A)} images (indices 0 to {split_point + overlap_count - 1})")
    print(f"Reconstruction B: {len(ids_B)} images (indices {split_point - overlap_count} to {n_images - 1})")
    print(f"Overlap: {overlap_count} images")
    
    # Create reconstruction A
    rec_A = pycolmap.Reconstruction()
    image_ids_A = set(ids_A)
    camera_ids_A = set()
    point3D_ids_A = set()
    
    # Add images and collect camera/point IDs for A
    for img_id in ids_A:
        image = reconstruction.images[img_id]
        rec_A.add_image(image)
        camera_ids_A.add(image.camera_id)
        
        for point2D in image.points2D:
            if point2D.has_point3D():
                point3D_ids_A.add(point2D.point3D_id)
    
    # Add cameras for A
    for cam_id in camera_ids_A:
        rec_A.add_camera(reconstruction.cameras[cam_id])
    
    # Add 3D points for A (only those visible in A's images)
    for pt_id in point3D_ids_A:
        point3D = reconstruction.points3D[pt_id]
        rec_A.add_point3D(point3D.xyz, pycolmap.Track(), point3D.color)
        
        # Add track elements only for images in A
        for element in point3D.track.elements:
            if element.image_id in image_ids_A:
                try:
                    rec_A.add_observation(pt_id, element.image_id, element.point2D_idx)
                except:
                    pass
    
    # Create reconstruction B
    rec_B = pycolmap.Reconstruction()
    image_ids_B = set(ids_B)
    camera_ids_B = set()
    point3D_ids_B = set()
    
    # Add images and collect camera/point IDs for B
    for img_id in ids_B:
        image = reconstruction.images[img_id]
        rec_B.add_image(image)
        camera_ids_B.add(image.camera_id)
        
        for point2D in image.points2D:
            if point2D.has_point3D():
                point3D_ids_B.add(point2D.point3D_id)
    
    # Add cameras for B
    for cam_id in camera_ids_B:
        rec_B.add_camera(reconstruction.cameras[cam_id])
    
    # Add 3D points for B (only those visible in B's images)
    for pt_id in point3D_ids_B:
        point3D = reconstruction.points3D[pt_id]
        rec_B.add_point3D(point3D.xyz, pycolmap.Track(), point3D.color)
        
        # Add track elements only for images in B
        for element in point3D.track.elements:
            if element.image_id in image_ids_B:
                try:
                    rec_B.add_observation(pt_id, element.image_id, element.point2D_idx)
                except:
                    pass
    
    print(f"\nReconstruction A stats:")
    print(f"  Cameras: {len(rec_A.cameras)}")
    print(f"  Images: {len(rec_A.images)}")
    print(f"  3D Points: {len(rec_A.points3D)}")
    
    print(f"\nReconstruction B stats:")
    print(f"  Cameras: {len(rec_B.cameras)}")
    print(f"  Images: {len(rec_B.images)}")
    print(f"  3D Points: {len(rec_B.points3D)}")
    
    # Calculate overlap statistics
    common_points = point3D_ids_A.intersection(point3D_ids_B)
    print(f"\nOverlap statistics:")
    print(f"  Common 3D points: {len(common_points)}")
    print(f"  Overlap ratio (points): {len(common_points) / len(point3D_ids_A.union(point3D_ids_B)):.2%}")
    
    return rec_A, rec_B


def main():
    parser = argparse.ArgumentParser(
        description="Split COLMAP reconstruction into two overlapping parts for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to input COLMAP reconstruction'
    )
    
    parser.add_argument(
        'output',
        type=str,
        help='Output directory (will create rec_A and rec_B subdirectories)'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.3,
        help='Overlap ratio (0.0 to 1.0). Default: 0.3'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['bin', 'txt'],
        default='bin',
        help='Output format. Default: bin'
    )
    
    args = parser.parse_args()
    
    # Validate overlap ratio
    if not 0.0 <= args.overlap <= 1.0:
        print(f"Error: Overlap ratio must be between 0.0 and 1.0, got {args.overlap}")
        return 1
    
    # Load reconstruction
    print(f"Loading reconstruction from: {args.input}")
    try:
        reconstruction = pycolmap.Reconstruction(args.input)
        print(f"Loaded: {len(reconstruction.cameras)} cameras, "
              f"{len(reconstruction.images)} images, "
              f"{len(reconstruction.points3D)} 3D points")
    except Exception as e:
        print(f"Error loading reconstruction: {e}")
        return 1
    
    # Split reconstruction
    print(f"\nSplitting with {args.overlap:.1%} overlap...")
    try:
        rec_A, rec_B = split_reconstruction(reconstruction, args.overlap)
    except Exception as e:
        print(f"Error splitting reconstruction: {e}")
        return 1
    
    # Save reconstructions
    output_dir = Path(args.output)
    output_A = output_dir / "rec_A"
    output_B = output_dir / "rec_B"
    
    print(f"\nSaving split reconstructions...")
    output_A.mkdir(parents=True, exist_ok=True)
    output_B.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.format == 'bin':
            rec_A.write_binary(str(output_A))
            rec_B.write_binary(str(output_B))
        else:
            rec_A.write_text(str(output_A))
            rec_B.write_text(str(output_B))
        
        print(f"Saved reconstruction A to: {output_A}")
        print(f"Saved reconstruction B to: {output_B}")
    except Exception as e:
        print(f"Error saving reconstructions: {e}")
        return 1
    
    print(f"\nYou can now test merging with:")
    print(f"python ESSN_merge_reconstruction.py {output_A} {output_B} {output_dir}/merged")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
