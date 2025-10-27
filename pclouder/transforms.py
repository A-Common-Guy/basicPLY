"""Transformation utilities for point cloud registration."""

import numpy as np
import open3d as o3d

def compute_normals(points, k=30):

    
    # Use Open3D for efficient normal computation (I'm a bit too lazy at this point to implement my own normal computation)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    
    # Orient normals consistently (toward viewpoint at origin)
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    return np.asarray(pcd.normals)


def compute_transformation(source_points, target_points, weights=None):

    if weights is None:
        weights = np.ones(source_points.shape[0])
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Compute weighted centroids
    source_centroid = np.sum(source_points * weights[:, np.newaxis], axis=0)
    target_centroid = np.sum(target_points * weights[:, np.newaxis], axis=0)
    
    # Center the points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Weighted covariance matrix
    H = (source_centered * weights[:, np.newaxis]).T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_centroid - R @ source_centroid
    
    # Build homogeneous transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    return transformation


def compute_transformation_point_to_plane(source_points, target_points, target_normals, weights=None):
    
    if weights is None:
        weights = np.ones(source_points.shape[0])
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Build the linear system for point-to-plane
    # We want to solve for the transformation parameters [alpha, beta, gamma, tx, ty, tz]
    # where alpha, beta, gamma are small rotation angles
    
    N = source_points.shape[0]
    A = np.zeros((N, 6))
    b = np.zeros(N)
    
    for i in range(N):
        s = source_points[i]  # source point
        d = target_points[i]  # destination (target) point
        n = target_normals[i]  # normal at destination
        w = np.sqrt(weights[i])  # weight
        
        # Cross product of source point and normal
        cross = np.cross(s, n)
        
        # Build row of A matrix: [cross, normal] weighted
        A[i] = w * np.hstack([cross, n])
        
        # Build b vector: dot product of (destination - source) and normal, weighted
        b[i] = w * np.dot(n, d - s)
    
    # Solve the linear system using least squares
    try:
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        # If solving fails, fall back to identity transform
        print("Warning: Point-to-plane solve failed, using identity")
        return np.eye(4)
    
    # Extract rotation angles and translation
    alpha, beta, gamma = params[0:3]  # rotation angles (small)
    tx, ty, tz = params[3:6]  # translation
    
    # Build rotation matrix from small angles (linearized)
    # For small angles: R ≈ I + [omega]_x where [omega]_x is the skew-symmetric matrix
    R = np.array([
        [1, -gamma, beta],
        [gamma, 1, -alpha],
        [-beta, alpha, 1]
    ])
    
    # Build homogeneous transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = [tx, ty, tz]
    
    return transformation


def apply_transformation(points, transformation):

    R = transformation[:3, :3]
    t = transformation[:3, 3]
    return points @ R.T + t


def generate_mirroring_matrices():

    mirroring_options = []
    
    for mirror_x in [1, -1]:
        for mirror_y in [1, -1]:
            for mirror_z in [1, -1]:
                matrix = np.eye(4)
                matrix[0, 0] = mirror_x
                matrix[1, 1] = mirror_y
                matrix[2, 2] = mirror_z
                
                mirrors = []
                if mirror_x == -1:
                    mirrors.append("X")
                if mirror_y == -1:
                    mirrors.append("Y")
                if mirror_z == -1:
                    mirrors.append("Z")
                
                description = f"Mirror: {', '.join(mirrors)}" if mirrors else "No mirroring"
                mirroring_options.append((description, matrix))
    
    return mirroring_options


def apply_mirroring(points, mirroring_matrix):

    return points @ mirroring_matrix[:3, :3].T

