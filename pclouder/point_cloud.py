"""Point cloud data management and preprocessing."""

import numpy as np
import open3d as o3d


class PointCloud:
    """Manages point cloud data with multi-resolution support."""
    
    def __init__(self, o3d_pcd):
        """
        Initialize a point cloud from Open3D point cloud.
        
        Args:
            o3d_pcd: Open3D PointCloud object
        """
        self.points = np.asarray(o3d_pcd.points)
        self.colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else None
        
        # Multi-resolution representations
        self._low_res = None
        self._medium_res = None
    
    @classmethod
    def from_file(cls, filepath):
        """Load point cloud from file."""
        pcd = o3d.io.read_point_cloud(filepath)
        return cls(pcd)
    
    @property
    def low_res(self):
        """Get low resolution version (voxel size: 20)."""
        if self._low_res is None:
            self._low_res = self._voxelize(20)
        return self._low_res
    
    @property
    def medium_res(self):
        """Get medium resolution version (voxel size: 10)."""
        if self._medium_res is None:
            self._medium_res = self._voxelize(10)
        return self._medium_res
    
    def _voxelize(self, voxel_size):
        """
        Downsample point cloud using voxel grid.
        
        Args:
            voxel_size: Size of voxels for downsampling
            
        Returns:
            Downsampled points as numpy array
        """
        if self.points.shape[0] == 0:
            return self.points
        
        min_bound = np.min(self.points, axis=0)
        voxel_indices = np.floor((self.points - min_bound) / voxel_size).astype(np.int32)
        
        voxel_dict = {}
        for i, point in enumerate(self.points):
            key = tuple(voxel_indices[i])
            if key not in voxel_dict:
                voxel_dict[key] = {'sum': point.copy(), 'count': 1}
            else:
                voxel_dict[key]['sum'] += point
                voxel_dict[key]['count'] += 1
        
        return np.array([v['sum'] / v['count'] for v in voxel_dict.values()])
    
    def to_o3d(self, points=None, color=None):
        """
        Convert to Open3D PointCloud object.
        
        Args:
            points: Optional custom points array (default: self.points)
            color: Optional uniform color [r, g, b] or color array
            
        Returns:
            Open3D PointCloud object
        """
        pcd = o3d.geometry.PointCloud()
        pts = points if points is not None else self.points
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        if color is not None:
            if isinstance(color, (list, tuple)) and len(color) == 3:
                pcd.paint_uniform_color(color)
            else:
                pcd.colors = o3d.utility.Vector3dVector(color)
        elif self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        
        return pcd
    
    def apply_transform(self, points, transformation):
        """
        Apply transformation to points.
        
        Args:
            points: Points array (N, 3)
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed points
        """
        R = transformation[:3, :3]
        t = transformation[:3, 3]
        return points @ R.T + t
    
    def __len__(self):
        return len(self.points)

