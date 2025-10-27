"""KD-Tree implementation for efficient spatial partitioning and nearest neighbor search."""

import numpy as np
from .utils import time_function


class Node:
    def __init__(self):
        self.point = None
        self.left = None
        self.right = None
        self.axis = None
        self.indices = None
    def set_point(self, point):
        # Accept either 1D (d,) or 2D (1,d) point arrays
        self.point = point
    def set_left(self, left):
        self.left = left
    def set_right(self, right):
        self.right = right
    def set_axis(self, axis):
        self.axis = axis
    def set_indices(self, indices):
        self.indices = indices

class KDTree:
    def __init__(self, leaf_size=128, dimension=3):
        self.root = None
        self.leaf_size = max(1, int(leaf_size))
        self.dimension = dimension

    @time_function
    def build(self, points, depth=0):
        # case no point, return none
        if points.shape[0] == 0:
            return None
        # case one point, return node with point
        if points.shape[0] == 1:
            newnode = Node()
            newnode.set_point(points[0])
            return newnode
        # split by axis cycling dimensions
        axis = depth % self.dimension
        # sort by the dimension axis we got
        sorted_indexes = np.argsort(points[:, axis])
        sorted_points = points[sorted_indexes]
        # median index and point
        median_index = sorted_indexes.shape[0] // 2
        median_point = sorted_points[median_index]
        # populate left and right side of the node
        newnode = Node()
        newnode.set_point(median_point)
        newnode.set_axis(axis)
        newnode.set_left(self.build(sorted_points[:median_index], depth + 1))
        newnode.set_right(self.build(sorted_points[median_index + 1 :], depth + 1))
        return newnode

    @time_function
    def build_optimized(self, points=None, depth=0, indices=None):
        # Initialize indices only at the top-level call
        if indices is None:
            pts = self.points if points is None else points
            if pts.shape[0] == 0:
                return None
            indices = np.arange(pts.shape[0], dtype=np.int64)
            # Keep a reference to the canonical points array
            if points is not None:
                self.points = points

        n_points = indices.shape[0]

        # No points
        if n_points == 0:
            return None

        # Leaf: store the indices to avoid creating millions of nodes
        if n_points <= self.leaf_size:
            leaf = Node()
            leaf.set_axis(depth % self.dimension)
            leaf.set_indices(indices)
            return leaf

        # Choose splitting axis
        axis = depth % self.dimension

        # Compute median position and in-place partition indices by the chosen axis
        median_index = n_points // 2
        # argpartition gives positions that would place kth in its final position
        order = np.argpartition(self.points[indices, axis], median_index)
        # Reorder this segment of indices in-place to avoid large copies
        indices[:] = indices[order]

        median_point_index = indices[median_index]

        node = Node()
        node.set_axis(axis)
        node.set_point(self.points[median_point_index])

        # Build subtrees using views (no copies) into the shared indices array
        left_view = indices[:median_index]
        right_view = indices[median_index+1:]

        node.set_left(self.build_optimized(depth=depth+1, indices=left_view))
        node.set_right(self.build_optimized(depth=depth+1, indices=right_view))
        return node



       
        
        
