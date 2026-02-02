#!/usr/bin/env python3
"""
COLMAP Utility Functions

Utility functions for reading and writing COLMAP reconstructions in text format.
Handles the specific format requirements that COLMAP expects.

COLMAP Text Format Reference:
    cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    images.txt:  IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (line 1)
                 X Y POINT3D_ID X Y POINT3D_ID ... (line 2, can be empty)
    points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK[]

Key Format Rules:
    - IDs are 0-based integers
    - Quaternions are in (qw, qx, qy, qz) order (w first)
    - Each image requires exactly 2 lines (data + POINTS2D)
    - Empty POINTS2D line is valid (just newline)
    - ERROR=-1 indicates no reprojection error computed
    - Comments start with # and appear at file start
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

try:
    import pycolmap
except ImportError:
    pycolmap = None


def write_cameras_txt(
    cameras: List[Tuple[int, 'pycolmap.Camera']],
    output_path: str
) -> None:
    """
    Write cameras.txt in COLMAP text format.
    
    Args:
        cameras: List of (camera_id, camera) tuples. IDs should be 0-based.
        output_path: Directory path to write cameras.txt
    
    Format:
        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # Number of cameras: N
        0 SIMPLE_RADIAL 960 720 338.0 480.0 360.0 0.0007
    """
    filepath = os.path.join(output_path, 'cameras.txt')
    with open(filepath, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id, camera in cameras:
            model_name = camera.model.name
            params_str = ' '.join(str(p) for p in camera.params)
            f.write(f"{cam_id} {model_name} {camera.width} {camera.height} {params_str}\n")


def write_images_txt(
    images: List[Dict],
    output_path: str,
    include_points2d: bool = False
) -> None:
    """
    Write images.txt in COLMAP text format.
    
    Args:
        images: List of image dicts with keys:
            - image_id: int (0-based)
            - qw, qx, qy, qz: float (quaternion, w first)
            - tx, ty, tz: float (translation)
            - camera_id: int (0-based)
            - name: str (image filename)
            - points2D: optional list of pycolmap.Point2D
        output_path: Directory path to write images.txt
        include_points2d: If True, write POINTS2D data; if False, write empty line
    
    Format:
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        # Number of images: N, mean observations per image: 0
        0 0.939 0.240 -0.240 -0.029 5.017 0.925 -2.169 0 image.jpg
        (empty line or POINTS2D data)
    """
    filepath = os.path.join(output_path, 'images.txt')
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        
        for img in images:
            # Line 1: Image data
            f.write(f"{img['image_id']} {img['qw']} {img['qx']} {img['qy']} {img['qz']} ")
            f.write(f"{img['tx']} {img['ty']} {img['tz']} {img['camera_id']} {img['name']}\n")
            
            # Line 2: POINTS2D (empty or with data)
            if include_points2d and 'points2D' in img and img['points2D']:
                points2d_str = ' '.join(
                    f"{p.xy[0]} {p.xy[1]} {p.point3D_id}" 
                    for p in img['points2D']
                )
                f.write(f"{points2d_str}\n")
            else:
                f.write("\n")  # Empty POINTS2D line required by COLMAP


def write_points3d_txt(
    points: List[Dict],
    output_path: str
) -> None:
    """
    Write points3D.txt in COLMAP text format.
    
    Args:
        points: List of point dicts with keys:
            - point3D_id: int (0-based)
            - xyz: array-like of 3 floats
            - color: array-like of 3 ints (RGB, 0-255)
            - error: float, optional (default -1)
            - track: list of (image_id, point2D_idx) tuples, optional
        output_path: Directory path to write points3D.txt
    
    Format:
        # 3D point list with one line of data per point:
        #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
        # Number of points: N, mean track length: 0
        0 24.25 -17.57 33.37 237 127 158 -1
    """
    filepath = os.path.join(output_path, 'points3D.txt')
    with open(filepath, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        
        for pt in points:
            xyz = pt['xyz']
            color = pt['color']
            error = pt.get('error', -1)
            
            f.write(f"{pt['point3D_id']} {xyz[0]} {xyz[1]} {xyz[2]} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])} {error}")
            
            # Write track if present
            if 'track' in pt and pt['track']:
                track_str = ' '.join(f"{img_id} {pt2d_idx}" for img_id, pt2d_idx in pt['track'])
                f.write(f" {track_str}")
            f.write("\n")


def write_reconstruction_txt(
    cameras: List[Tuple[int, 'pycolmap.Camera']],
    images: List[Dict],
    points: List[Dict],
    output_path: str,
    include_points2d: bool = False
) -> None:
    """
    Write complete COLMAP reconstruction in text format.
    
    Args:
        cameras: List of (camera_id, camera) tuples
        images: List of image dicts (see write_images_txt)
        points: List of point dicts (see write_points3d_txt)
        output_path: Directory to write reconstruction files
        include_points2d: Whether to include POINTS2D in images.txt
    """
    os.makedirs(output_path, exist_ok=True)
    write_cameras_txt(cameras, output_path)
    write_images_txt(images, output_path, include_points2d)
    write_points3d_txt(points, output_path)


def pycolmap_image_to_dict(
    image: 'pycolmap.Image',
    new_image_id: int,
    new_camera_id: int
) -> Dict:
    """
    Convert pycolmap Image to dict format for writing.
    
    Args:
        image: pycolmap.Image object
        new_image_id: New 0-based image ID
        new_camera_id: New 0-based camera ID
    
    Returns:
        Dict with image data ready for write_images_txt
    """
    pose = image.cam_from_world()
    quat = pose.rotation.quat  # xyzw format from pycolmap
    trans = pose.translation
    
    return {
        'image_id': new_image_id,
        'qw': quat[3],  # COLMAP uses w first
        'qx': quat[0],
        'qy': quat[1],
        'qz': quat[2],
        'tx': trans[0],
        'ty': trans[1],
        'tz': trans[2],
        'camera_id': new_camera_id,
        'name': image.name,
        'points2D': image.points2D
    }


def pycolmap_point3d_to_dict(
    point: 'pycolmap.Point3D',
    new_point_id: int,
    image_id_map: Optional[Dict[int, int]] = None
) -> Dict:
    """
    Convert pycolmap Point3D to dict format for writing.
    
    Args:
        point: pycolmap.Point3D object
        new_point_id: New 0-based point ID
        image_id_map: Optional mapping from old to new image IDs for track remapping
    
    Returns:
        Dict with point data ready for write_points3d_txt
    """
    result = {
        'point3D_id': new_point_id,
        'xyz': point.xyz,
        'color': point.color,
        'error': -1  # Default to -1 (no error computed)
    }
    
    # Optionally include remapped track
    if image_id_map is not None:
        track = []
        for te in point.track.elements:
            if te.image_id in image_id_map:
                track.append((image_id_map[te.image_id], te.point2D_idx))
        if track:
            result['track'] = track
            result['error'] = point.error
    
    return result


def merge_reconstructions_to_colmap_txt(
    reconstruction_A: 'pycolmap.Reconstruction',
    reconstruction_B: 'pycolmap.Reconstruction',
    output_path: str,
    skip_duplicate_images: bool = True
) -> Tuple[int, int, int]:
    """
    Merge two pycolmap reconstructions and write as COLMAP text format.
    
    Assumes reconstruction_A has already been transformed to align with B.
    Uses 0-based IDs and ignores tracks for simplicity.
    
    Args:
        reconstruction_A: First reconstruction (already aligned)
        reconstruction_B: Second reconstruction (target frame)
        output_path: Directory to write merged reconstruction
        skip_duplicate_images: If True, skip images from A that exist in B
    
    Returns:
        Tuple of (num_cameras, num_images, num_points)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Build set of existing image names in B
    existing_names = {img.name for img in reconstruction_B.images.values()}
    
    # === Merge cameras (0-based IDs) ===
    cameras = []
    camera_id_map_B = {}
    camera_id_map_A = {}
    new_cam_id = 0
    
    for old_id, camera in reconstruction_B.cameras.items():
        camera_id_map_B[old_id] = new_cam_id
        cameras.append((new_cam_id, camera))
        new_cam_id += 1
    
    for old_id, camera in reconstruction_A.cameras.items():
        camera_id_map_A[old_id] = new_cam_id
        cameras.append((new_cam_id, camera))
        new_cam_id += 1
    
    # === Merge images (0-based IDs) ===
    images = []
    new_img_id = 0
    
    for img_id, image in reconstruction_B.images.items():
        images.append(pycolmap_image_to_dict(image, new_img_id, camera_id_map_B[image.camera_id]))
        new_img_id += 1
    
    for img_id, image in reconstruction_A.images.items():
        if skip_duplicate_images and image.name in existing_names:
            continue
        images.append(pycolmap_image_to_dict(image, new_img_id, camera_id_map_A[image.camera_id]))
        new_img_id += 1
    
    # === Merge 3D points (0-based IDs, no tracks) ===
    points = []
    new_pt_id = 0
    
    for pt_id, point in reconstruction_B.points3D.items():
        points.append({
            'point3D_id': new_pt_id,
            'xyz': point.xyz,
            'color': point.color
        })
        new_pt_id += 1
    
    for pt_id, point in reconstruction_A.points3D.items():
        points.append({
            'point3D_id': new_pt_id,
            'xyz': point.xyz,
            'color': point.color
        })
        new_pt_id += 1
    
    # === Write files ===
    write_reconstruction_txt(cameras, images, points, output_path)
    
    return len(cameras), len(images), len(points)
