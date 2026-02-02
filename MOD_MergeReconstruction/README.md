# ESSN Merge Reconstruction

Standalone script to register and merge two COLMAP reconstructions using the ColabSfM registration method.

## Overview

This script:
1. Loads two COLMAP reconstructions (A and B)
2. Extracts 3D point clouds and camera viewpoints from both
3. Uses ColabSfM's deep learning model to find correspondences
4. Estimates the transformation matrix (SE(3) or Sim(3)) using RANSAC
5. Transforms reconstruction A to align with reconstruction B's coordinate frame
6. Merges both reconstructions into a single unified model
7. Saves the merged reconstruction to disk

## Usage

### Basic Usage

```bash
python ESSN_merge_reconstruction.py <reconstruction_A_path> <reconstruction_B_path> <output_path>
```

### Examples

**SE(3) Registration (rigid transformation):**
```bash
# Merge two different reconstructions of the same scene
python ESSN_merge_reconstruction.py \
    /path/to/scene/reconstruction_1 \
    /path/to/scene/reconstruction_2 \
    /path/to/output/merged \
    --mode se3
```

**Sim(3) Registration (with scale):**
```bash
# Use sim3 if reconstructions have different scales
python ESSN_merge_reconstruction.py \
    /path/to/reconstruction_phone \
    /path/to/reconstruction_camera \
    /path/to/output/merged \
    --mode sim3
```

**Save transformation matrix:**
```bash
python ESSN_merge_reconstruction.py \
    /data/rec_A \
    /data/rec_B \
    /data/merged \
    --save-transform
```

**Use custom weights:**
```bash
python ESSN_merge_reconstruction.py \
    /data/rec_A \
    /data/rec_B \
    /data/merged \
    --weights /path/to/custom_weights.pth
```

**Save as text format:**
```bash
python ESSN_merge_reconstruction.py \
    /data/rec_A \
    /data/rec_B \
    /data/merged \
    --format txt
```

## Arguments

### Required Arguments

- `reconstruction_A`: Path to first COLMAP reconstruction (will be transformed to align with B)
- `reconstruction_B`: Path to second COLMAP reconstruction (serves as the target coordinate frame)
- `output`: Path where the merged reconstruction will be saved

### Optional Arguments

- `--mode {se3,sim3}`: Registration mode
  - `se3` (default): Rigid transformation (rotation + translation)
  - `sim3`: Similarity transformation (rotation + translation + scale)

- `--weights PATH`: Path to custom model weights
  - If not specified, downloads default pre-trained weights automatically

- `--format {bin,txt}`: Output format for merged reconstruction
  - `bin` (default): Binary format (more compact, faster to load)
  - `txt`: Text format (human-readable)

- `--save-transform`: Save the 4x4 transformation matrix as `transformation.npy`

## Requirements

- Python 3.8+
- PyTorch
- pycolmap
- numpy
- ColabSfM (parent directory)

Make sure you're using the correct conda/micromamba environment:
```bash
micromamba activate vlm3r
```

## Input Format

The input reconstructions should be in standard COLMAP format:
```
reconstruction_A/
  cameras.bin (or cameras.txt)
  images.bin (or images.txt)
  points3D.bin (or points3D.txt)
```

## Output Format

The merged reconstruction is saved in the same COLMAP format:
```
output/
  cameras.bin (or cameras.txt)
  images.bin (or images.txt)
  points3D.bin (or points3D.txt)
  [transformation.npy]  # if --save-transform is used
```

## Notes

- The script handles ID conflicts automatically by renumbering cameras, images, and 3D points as needed
- Reconstruction B serves as the reference frame; A is transformed to align with B
- The merged reconstruction contains all cameras, images, and 3D points from both reconstructions
- For best results, the two reconstructions should have some overlap in their scene coverage

## Real-World Example

Suppose you have two separate reconstructions of the same building:
- One from the ground floor: `/data/building/ground_floor`
- One from the first floor: `/data/building/first_floor`

```bash
cd MOD_MergeReconstruction

# Activate the environment
micromamba activate vlm3r

# Merge the two reconstructions
python ESSN_merge_reconstruction.py \
    /data/building/ground_floor \
    /data/building/first_floor \
    /data/building/merged_full_building \
    --mode se3 \
    --save-transform
```

This will:
1. Register the ground floor to the first floor's coordinate system
2. Merge all cameras, images, and 3D points into one unified reconstruction
3. Save the result in `/data/building/merged_full_building`
4. Save the transformation matrix as `transformation.npy`

## Technical Details

### Registration Pipeline

1. **Point Cloud Extraction**: Extracts 3D points and camera viewpoints from both reconstructions
2. **Feature Extraction**: Uses RoITr deep learning model with rotation-invariant features
3. **Correspondence Matching**: Finds corresponding points between the two point clouds
4. **Transformation Estimation**: Uses Open3D RANSAC to robustly estimate the transformation
5. **Transformation Application**: Uses pycolmap's built-in `transform()` method
6. **Reconstruction Merging**: Merges all elements with automatic ID conflict resolution

### Transformation Matrix

The transformation matrix T is a 4×4 homogeneous matrix that transforms points from reconstruction A to reconstruction B's coordinate frame:

```
p_B = T @ p_A

where T = [R  t]
          [0  1]
```

For SE(3): R is a 3×3 rotation matrix, t is a 3×1 translation vector
For Sim(3): R includes scale: R = s * R_rot

## Troubleshooting

**CUDA out of memory:**
- The script automatically downsamples point clouds larger than 30,000 points
- If issues persist, you can reduce GPU memory usage by processing smaller reconstructions

**Low number of matches:**
- Ensure the reconstructions have sufficient overlap
- Try using `sim3` mode if there are scale differences
- Check that the reconstructions are valid and contain 3D points

**Import errors:**
- Make sure ColabSfM is in the parent directory or adjust `sys.path` in the script
- Verify all dependencies are installed in your environment
