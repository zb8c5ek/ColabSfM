"""
Umeyama Algorithm for SIM3 (Similarity Transformation) Estimation

This module implements the Umeyama algorithm for estimating similarity transformations
(rotation, translation, and scale) between two sets of 3D points, along with a 
RANSAC-based robust version for handling outliers.

References:
    - Umeyama, S. (1991). "Least-squares estimation of transformation parameters 
      between two point patterns." IEEE TPAMI, 13(4), 376-380.

Author: Xuan-Li Vons CHEN
Date: February 2, 2026
"""

import numpy as np
import math
from typing import Tuple, Optional


def umeyama(ptsA: np.ndarray, ptsB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the similarity transformation (rotation, translation, scale) between two point sets
    using the Umeyama algorithm (closed-form solution).
    
    The transformation is computed such that: ptsB ≈ scale * (ptsA @ R.T) + t
    
    Args:
        ptsA (np.ndarray): Source points, shape (N, 3)
        ptsB (np.ndarray): Target points, shape (N, 3)
    
    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - R (np.ndarray): Rotation matrix, shape (3, 3)
            - translation (np.ndarray): Translation vector, shape (3,)
            - scale (float): Scale factor
    
    Example:
        >>> ptsA = np.random.rand(100, 3)
        >>> scale_true = 2.5
        >>> R_true = np.eye(3)
        >>> t_true = np.array([1, 2, 3])
        >>> ptsB = scale_true * (ptsA @ R_true.T) + t_true
        >>> R, t, s = umeyama(ptsA, ptsB)
        >>> print(f"Estimated scale: {s:.4f}, True scale: {scale_true}")
    """
    # Assume array shape is Nx3
    n = ptsA.shape[0]
    
    if n < 3:
        raise ValueError(f"At least 3 point correspondences required, got {n}")
    
    # Compute centroids
    centerA = np.mean(ptsA, axis=0)
    centerB = np.mean(ptsB, axis=0)
    
    # Remove center point (mean-centering)
    pt3DCenteredA = ptsA - centerA.reshape(1, -1)
    pt3DCenteredB = ptsB - centerB.reshape(1, -1)
    
    # Compute covariance matrix
    Sxy = (pt3DCenteredB.T @ pt3DCenteredA) / n
    
    # Get rotation via SVD
    S = np.eye(3)
    if np.linalg.det(Sxy) < 0:
        S[2, 2] = -1
    U, D, Vh = np.linalg.svd(Sxy)
    R = U @ S @ Vh
    
    # Get scale
    sigmaX = np.mean(np.sum(pt3DCenteredA * pt3DCenteredA, axis=1))
    scale = np.trace(np.diag(D) @ S) / sigmaX
    
    # Get translation
    translation = centerB - scale * (R @ centerA)
    
    return R, translation, scale


def ransac_umeyama(
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    npoints: int = 3,
    threshold: float = 0.05,
    max_iter: int = 10000,
    compute_rmse: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, ...]:
    """
    RANSAC-based robust SIM3 estimation using the Umeyama algorithm.
    
    Iteratively samples minimal sets of point correspondences and estimates
    the similarity transformation, keeping the solution with the most inliers.
    
    Args:
        ptsA (np.ndarray): Source points, shape (N, 3)
        ptsB (np.ndarray): Target points, shape (N, 3)
        npoints (int): Number of points in minimal sample set (default: 3)
        threshold (float): Inlier distance threshold (default: 0.05)
        max_iter (int): Maximum number of RANSAC iterations (default: 10000)
        compute_rmse (bool): Whether to compute RMSE metrics (default: False)
    
    Returns:
        If compute_rmse is False:
            Tuple[np.ndarray, np.ndarray, float]:
                - R_sol (np.ndarray): Best rotation matrix, shape (3, 3)
                - t_sol (np.ndarray): Best translation vector, shape (3,)
                - s_sol (float): Best scale factor
        
        If compute_rmse is True:
            Tuple[np.ndarray, np.ndarray, float, float, float]:
                - R_sol (np.ndarray): Best rotation matrix, shape (3, 3)
                - t_sol (np.ndarray): Best translation vector, shape (3,)
                - s_sol (float): Best scale factor
                - inlierRMSE (float): RMSE of inliers only
                - rmse (float): Overall RMSE
    
    Example:
        >>> # Create noisy correspondences
        >>> ptsA = np.random.rand(100, 3)
        >>> scale_true = 1.5
        >>> R_true = np.eye(3)
        >>> t_true = np.array([0.5, 1.0, 1.5])
        >>> ptsB = scale_true * (ptsA @ R_true.T) + t_true + np.random.randn(100, 3) * 0.01
        >>> R, t, s = ransac_umeyama(ptsA, ptsB, threshold=0.1)
        >>> print(f"Scale: {s:.4f}, Expected: {scale_true}")
    """
    if ptsA.shape[0] < npoints:
        raise ValueError(f"Not enough points: {ptsA.shape[0]} < {npoints}")
    
    # Initialize solution
    R_sol = np.eye(3)
    t_sol = np.zeros(3)
    s_sol = 0
    num_inliers = 0
    inlier_ratio = 0
    maxIter = max_iter
    iter = 0
    
    if compute_rmse:
        inlierRMSE = np.inf
        rmse = np.inf
    
    while iter < maxIter:
        # Select randomly npoints correspondences
        sel_idx = np.random.choice(ptsB.shape[0], npoints, replace=False)
        
        # Compute model using Umeyama algorithm
        try:
            rot, trans, scale = umeyama(ptsA[sel_idx, :], ptsB[sel_idx])
        except (np.linalg.LinAlgError, ValueError):
            iter += 1
            continue
        
        # Evaluate model on all points
        ptsAB = scale * (ptsA @ rot.T) + trans
        dists = np.linalg.norm(ptsAB - ptsB, axis=1)
        inlier_mask = (dists < threshold)
        nin = inlier_mask.sum()
        
        # Update best solution if more inliers found
        if nin > num_inliers:
            if compute_rmse:
                inlierRMSE = np.sqrt(np.mean(dists[inlier_mask]**2))
                rmse = np.sqrt(np.mean(dists**2))
            
            num_inliers = nin
            inlier_ratio = num_inliers / ptsB.shape[0]
            R_sol = rot
            t_sol = trans
            s_sol = scale
            
            # Adaptive RANSAC: update max iterations based on inlier ratio
            maxIter = math.ceil(
                min(max_iter, np.log(1 - 0.99) / np.log(1 - inlier_ratio**npoints))
            )
        
        iter += 1
    
    if compute_rmse:
        return R_sol, t_sol, s_sol, inlierRMSE, rmse
    else:
        return R_sol, t_sol, s_sol


def apply_sim3_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Apply SIM3 transformation to a set of points.
    
    Args:
        points (np.ndarray): Input points, shape (N, 3)
        rotation (np.ndarray): Rotation matrix, shape (3, 3)
        translation (np.ndarray): Translation vector, shape (3,)
        scale (float): Scale factor
    
    Returns:
        np.ndarray: Transformed points, shape (N, 3)
    
    Example:
        >>> points = np.random.rand(50, 3)
        >>> R = np.eye(3)
        >>> t = np.array([1, 2, 3])
        >>> s = 2.0
        >>> transformed = apply_sim3_transform(points, R, t, s)
    """
    return scale * (points @ rotation.T) + translation


def compute_sim3_error(
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float
) -> Tuple[float, float, int]:
    """
    Compute error metrics for a SIM3 transformation.
    
    Args:
        ptsA (np.ndarray): Source points, shape (N, 3)
        ptsB (np.ndarray): Target points, shape (N, 3)
        rotation (np.ndarray): Rotation matrix, shape (3, 3)
        translation (np.ndarray): Translation vector, shape (3,)
        scale (float): Scale factor
    
    Returns:
        Tuple[float, float, int]:
            - rmse (float): Root mean square error
            - mean_error (float): Mean error
            - num_points (int): Number of points evaluated
    """
    ptsA_transformed = apply_sim3_transform(ptsA, rotation, translation, scale)
    errors = np.linalg.norm(ptsA_transformed - ptsB, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    
    return rmse, mean_error, len(ptsA)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 70)
    print("Umeyama SIM3 Solver - Example Usage")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_points = 100
    ptsA = np.random.rand(n_points, 3) * 10
    
    # Ground truth transformation
    scale_gt = 2.5
    angle = np.pi / 6  # 30 degrees
    R_gt = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_gt = np.array([5.0, 3.0, 2.0])
    
    # Create transformed points with noise
    ptsB = scale_gt * (ptsA @ R_gt.T) + t_gt
    noise = np.random.randn(n_points, 3) * 0.01
    ptsB += noise
    
    # Add some outliers
    n_outliers = 10
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    ptsB[outlier_indices] += np.random.randn(n_outliers, 3) * 5
    
    print(f"\nGround truth:")
    print(f"  Scale: {scale_gt:.4f}")
    print(f"  Translation: {t_gt}")
    
    # Test Umeyama (without outliers)
    print(f"\n{'='*70}")
    print("1. Umeyama Algorithm (all points, includes outliers)")
    print(f"{'='*70}")
    R_est, t_est, s_est = umeyama(ptsA, ptsB)
    print(f"  Estimated scale: {s_est:.4f}")
    print(f"  Estimated translation: {t_est}")
    rmse, mean_error, _ = compute_sim3_error(ptsA, ptsB, R_est, t_est, s_est)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean error: {mean_error:.4f}")
    
    # Test RANSAC Umeyama
    print(f"\n{'='*70}")
    print("2. RANSAC + Umeyama (robust to outliers)")
    print(f"{'='*70}")
    R_ransac, t_ransac, s_ransac, inlier_rmse, total_rmse = ransac_umeyama(
        ptsA, ptsB, npoints=3, threshold=0.1, max_iter=1000, compute_rmse=True
    )
    print(f"  Estimated scale: {s_ransac:.4f}")
    print(f"  Estimated translation: {t_ransac}")
    print(f"  Inlier RMSE: {inlier_rmse:.4f}")
    print(f"  Total RMSE: {total_rmse:.4f}")
    
    # Compute final errors
    ptsA_transformed = apply_sim3_transform(ptsA, R_ransac, t_ransac, s_ransac)
    final_errors = np.linalg.norm(ptsA_transformed - ptsB, axis=1)
    inliers = final_errors < 0.1
    print(f"  Number of inliers: {inliers.sum()} / {n_points}")
    print(f"  Inlier ratio: {inliers.sum() / n_points:.2%}")
    
    print(f"\n{'='*70}")
    print("Testing complete!")
    print(f"{'='*70}\n")
