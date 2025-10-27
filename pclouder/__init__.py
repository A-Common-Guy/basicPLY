"""
PClouder - Point Cloud Registration using Iterative Closest Point (ICP)

A robust point cloud registration library featuring:
- Custom KD-Tree implementation for efficient nearest neighbor search
- Multi-resolution ICP with automatic mirroring detection
- Robust loss functions for outlier handling
- Parallel processing support
"""

from .icp import ICPRegistration
from .kdtree import KDTree
from .point_cloud import PointCloud
from .transforms import compute_normals
from .visualization import plot_convergence, plot_loss_comparison

__version__ = "1.0.0"
__all__ = ["ICPRegistration", "KDTree", "PointCloud", "compute_normals", 
           "plot_convergence", "plot_loss_comparison"]

