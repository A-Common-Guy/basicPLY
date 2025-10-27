"""General utility functions."""

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
        if not hasattr(wrapper, '_in_call'):
            wrapper._in_call = False
        
        if not wrapper._in_call:
            wrapper._in_call = True
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"{func.__name__} took {elapsed:.6f} seconds")
                return result
            finally:
                wrapper._in_call = False
        else:
            return func(*args, **kwargs)
    
    return wrapper


def nearest_neighbor_search(query_point, root, points_array):
    """
    Iterative nearest neighbor search in KD-tree.
        
    Args:
        query_point: Point to find the nearest neighbor for
        root: Root node of the KD-tree
        points_array: Numpy array of points
    
    Returns:
        Tuple of (nearest_point, distance)
    """
    stack = [root]
    best = (None, np.inf)

    while stack:
        node = stack.pop()
        if node is None:
            continue

        # Leaf node: check all points in the leaf
        if node.indices is not None:
            leaf_points = points_array[node.indices]
            dists = np.linalg.norm(leaf_points - query_point, axis=1)
            idx = np.argmin(dists)
            dist = dists[idx]
            if dist < best[1]:
                best = (leaf_points[idx], dist)
            continue

        # Internal node: check node point
        dist = np.linalg.norm(node.point - query_point)
        if dist < best[1]:
            best = (node.point, dist)

        # Traverse tree
        axis = node.axis
        if query_point[axis] < node.point[axis]:
            near_node, far_node = node.left, node.right
        else:
            near_node, far_node = node.right, node.left
        
        stack.append(near_node)
        if abs(query_point[axis] - node.point[axis]) < best[1]:
            stack.append(far_node)
    
    return best
