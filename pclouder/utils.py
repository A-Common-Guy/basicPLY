import time
from functools import wraps
import numpy as np


def time_function(func):
    """
    Decorator to time function execution. 
    For recursive functions, only times the top-level call.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're already in a recursive call
        if not hasattr(wrapper, '_in_call'):
            wrapper._in_call = False
        
        # Only time the top-level call
        if not wrapper._in_call:
            wrapper._in_call = True
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"{func.__name__} took {elapsed:.6f} seconds")
                return result
            finally:
                wrapper._in_call = False
        else:
            # We're in a recursive call, just execute without timing
            return func(*args, **kwargs)
    
    return wrapper


def voxelize_point_cloud(points, voxel_size):

    if points.shape[0] == 0:
        return points
    
    # Get the minimum corner of the bounding box
    min_bound = np.min(points, axis=0)
    
    # Compute voxel indices for each point
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    
    # Use dictionary to accumulate points per voxel
    voxel_dict = {}
    
    for i in range(len(points)):
        # Create a tuple key from voxel indices
        key = tuple(voxel_indices[i])
        
        if key not in voxel_dict:
            voxel_dict[key] = {'sum': points[i].copy(), 'count': 1}
        else:
            voxel_dict[key]['sum'] += points[i]
            voxel_dict[key]['count'] += 1
    
    # Compute centroids
    downsampled_points = np.array([v['sum'] / v['count'] for v in voxel_dict.values()])
    
    return downsampled_points


def compute_transformation(source_points, target_points):
    #compute the transformation between the two point clouds
    #start by matching the centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    #center the points
    source_points_centered = source_points - source_centroid
    target_points_centered = target_points - target_centroid

    #get covariance matrix
    H = source_points_centered.T @ target_points_centered
    U, S, Vt = np.linalg.svd(H)
    #compute the rotation matrix
    R = Vt.T @ U.T
    #handle the case where the determinant is negative (symmetry breaking)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T  # Recompute R with corrected Vt
    t = target_centroid - R @ source_centroid
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    return transformation_matrix

def apply_transformation(points, transformation_matrix):
    # Points are (N, 3), we need to apply R @ points.T then transpose back
    return points @ transformation_matrix[:3, :3].T + transformation_matrix[:3, 3]



def distance(point1, point2):
    #dont need to square root because we are only comparing distances
    return np.sum((point1 - point2) ** 2)


def _nearest_neighbor_search_iterative(query_point, root, points_array):
    """
    Standalone function for nearest neighbor search that can be pickled for parallel processing.
    
    Args:
        query_point: The point to find the nearest neighbor for
        root: The root node of the KD-tree
        points_array: The numpy array of points (not an Open3D object)
    
    Returns:
        Tuple of (nearest_point, distance)
    """
    stack = [root]
    best = (None, np.inf)

    while stack:
        node = stack.pop()
        if node is None:
            continue

        if node.indices is not None:
            leaf_points = points_array[node.indices]
            dists = np.linalg.norm(leaf_points - query_point, axis=1)
            idx = np.argmin(dists)
            dist = dists[idx]
            if dist < best[1]:
                best = (leaf_points[idx], dist)
            continue

        dist = np.linalg.norm(node.point - query_point)
        if dist < best[1]:
            best = (node.point, dist)

        axis = node.axis
        near_node, far_node = (node.left, node.right) if query_point[axis] < node.point[axis] else (node.right, node.left)
        stack.append(near_node)  # Always explore the near side first
        if abs(query_point[axis] - node.point[axis]) < best[1]:
            stack.append(far_node)  # Only explore far side if hypersphere intersects
    return best


def generate_mirroring_matrices():
    """
    Generate all 8 possible mirroring transformation matrices.
    
    Returns:
        List of tuples: [(description, transformation_matrix), ...]
        Each matrix represents a mirroring along X, Y, Z axes or combinations.
    """
    mirroring_options = []
    
    # Generate all combinations: 2^3 = 8 options
    for mirror_x in [1, -1]:
        for mirror_y in [1, -1]:
            for mirror_z in [1, -1]:
                matrix = np.eye(4)
                matrix[0, 0] = mirror_x
                matrix[1, 1] = mirror_y
                matrix[2, 2] = mirror_z
                
                # Create a description
                mirrors = []
                if mirror_x == -1:
                    mirrors.append("X")
                if mirror_y == -1:
                    mirrors.append("Y")
                if mirror_z == -1:
                    mirrors.append("Z")
                
                if mirrors:
                    description = f"Mirror: {', '.join(mirrors)}"
                else:
                    description = "No mirroring"
                
                mirroring_options.append((description, matrix))
    
    return mirroring_options


def apply_mirroring(points, mirroring_matrix):
    """
    Apply a mirroring transformation to a set of points.
    
    Args:
        points: (N, 3) array of points
        mirroring_matrix: 4x4 transformation matrix
    
    Returns:
        (N, 3) array of mirrored points
    """
    return points @ mirroring_matrix[:3, :3].T

