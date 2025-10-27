"""Iterative Closest Point (ICP) algorithm implementation."""

import numpy as np
import open3d as o3d
from joblib import Parallel, delayed
import pickle
import os
import time

from .kdtree import KDTree
from .point_cloud import PointCloud
from .transforms import (compute_transformation, compute_transformation_point_to_plane,
                         apply_transformation, compute_normals,
                         generate_mirroring_matrices, apply_mirroring as apply_mirroring_transform)
from .losses import get_loss_function
from .utils import time_function, nearest_neighbor_search


class ICPRegistration:
    """
    ICP registration with multi-resolution and automatic mirroring detection.
    """
    
    def __init__(self, source, target):
        """
        Initialize ICP registration.
        
        Args:
            source: PointCloud object or path to source point cloud file
            target: PointCloud object or path to target point cloud file
        """
        if isinstance(source, str):
            source = PointCloud.from_file(source)
        if isinstance(target, str):
            target = PointCloud.from_file(target)
        
        self.source = source
        self.target = target
        self.initial_transform = np.eye(4)
    
    def register(self, max_iterations=100, quality="medium", 
                 loss_fn='none', loss_params=None, 
                 visualize=False, target_distance=None,
                 apply_mirroring=None, method='point_to_point'):
        """
        Run ICP registration.
        
        Args:
            max_iterations: Maximum number of iterations
            quality: Resolution quality: 'full', 'medium', or 'low'
            loss_fn: Loss function: 'none', 'huber', 'tukey', 'percentile'
            loss_params: Dictionary of loss function parameters
            visualize: Whether to visualize registration progress
            target_distance: Stop when mean distance falls below this
            apply_mirroring: Optional mirroring to apply to source before ICP.
                           Can be: 'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ', or None
            method: ICP method: 'point_to_point' (default) or 'point_to_plane'
        
        Returns:
            Tuple of (transformation, mean_distances, intermediate_transforms)
        """
        # Start timing
        total_start = time.time()
        
        # Select point resolution
        source_points = self._get_points(self.source, quality)
        target_points = self._get_points(self.target, quality)
        
        print(f"\n{'='*70}")
        print(f"RUNTIME BENCHMARKS - {quality.upper()} QUALITY")
        print(f"{'='*70}")
        print(f"ICP Method: {method.upper().replace('_', '-')}")
        print(f"Source points: {len(source_points):,}")
        print(f"Target points: {len(target_points):,}")
        
        # Apply mirroring if specified
        mirroring_matrix = None
        if apply_mirroring:
            mirror_start = time.time()
            mirroring_matrix = self._get_mirroring_matrix(apply_mirroring)
            source_points = apply_mirroring_transform(source_points, mirroring_matrix)
            mirror_time = time.time() - mirror_start
            print(f"Applied mirroring: {apply_mirroring} ({mirror_time:.3f}s)")
        
        # Build KD-tree for target
        print(f"\nBuilding KD-tree...")
        tree_start = time.time()
        tree = KDTree(leaf_size=128, dimension=3)
        root = tree.build_optimized(target_points)
        tree_time = time.time() - tree_start
        print(f"  ✓ KD-tree built in {tree_time:.3f}s")
        
        # Compute normals for point-to-plane ICP
        target_normals = None
        if method == 'point_to_plane':
            print(f"\nComputing target normals...")
            normals_start = time.time()
            target_normals = compute_normals(target_points, k=30)
            normals_time = time.time() - normals_start
            print(f"  ✓ Normals computed in {normals_time:.3f}s")
        
        # Get loss function
        loss_func, loss_params_final = get_loss_function(loss_fn, loss_params)
        
        # Initialize tracking
        mean_distances = []
        intermediate_transforms = [self.initial_transform.copy()]
        cumulative_transform = self.initial_transform.copy()
        
        # Setup visualization
        vis, source_vis, target_vis = None, None, None
        if visualize:
            vis, source_vis, target_vis = self._setup_visualization(
                source_points, target_points, loss_fn
            )
        
        # ICP iterations
        print(f"\nStarting ICP iterations...")
        print(f"{'─'*70}")
        iteration_times = []
        correspondence_times = []
        transform_times = []
        
        for i in range(max_iterations):
            iter_start = time.time()
            
            # Transform source points
            transform_start = time.time()
            source_transformed = apply_transformation(source_points, cumulative_transform)
            transform_time = time.time() - transform_start
            
            # Find nearest neighbors
            corr_start = time.time()
            neighbors, distances = self._find_correspondences(
                source_transformed, root, target_points
            )
            corr_time = time.time() - corr_start
            correspondence_times.append(corr_time)
            
            # Compute weights for robust estimation
            weights = None
            if loss_func is not None:
                weights = loss_func(distances, **loss_params_final)
                n_outliers = np.sum(weights < 0.1)
                if i % 10 == 0:
                    print(f"  Iteration {i}: {n_outliers}/{len(weights)} outliers filtered")
            
            mean_distance = np.mean(distances)
            mean_distances.append(mean_distance)
            
            # Compute transformation update
            transform_comp_start = time.time()
            
            if method == 'point_to_plane':
                # Get normals for the corresponding target points
                # Find indices of neighbors in target_points
                neighbor_indices = np.array([
                    np.where((target_points == n).all(axis=1))[0][0] 
                    for n in neighbors
                ])
                corresponding_normals = target_normals[neighbor_indices]
                
                transformation = compute_transformation_point_to_plane(
                    source_transformed, neighbors, corresponding_normals, weights=weights
                )
            else:
                transformation = compute_transformation(
                    source_transformed, neighbors, weights=weights
                )
            
            transform_comp_time = time.time() - transform_comp_start
            transform_times.append(transform_comp_time)
            
            cumulative_transform = transformation @ cumulative_transform
            intermediate_transforms.append(cumulative_transform.copy())
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            # Print iteration info every 10 iterations
            if i % 10 == 0:
                print(f"Iter {i:3d}: distance={mean_distance:.4f} | "
                      f"corr={corr_time:.3f}s | total={iter_time:.3f}s")
            
            # Update visualization
            if visualize and vis is not None:
                source_vis.points = o3d.utility.Vector3dVector(source_transformed)
                vis.update_geometry(source_vis)
                vis.poll_events()
                vis.update_renderer()
            
            # Check convergence
            if target_distance is not None and mean_distance < target_distance:
                print(f"\n✓ Reached target distance ({target_distance}) at iteration {i+1}")
                break
            
            if i > 0 and mean_distances[i-1] - mean_distances[i] < 0.001:
                print(f"\n✓ Converged at iteration {i}")
                break
        
        if visualize and vis is not None:
            vis.destroy_window()
        
        # Calculate total time
        total_time = time.time() - total_start
        actual_iterations = len(mean_distances)
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"Total runtime:           {total_time:.3f}s")
        print(f"KD-tree build time:      {tree_time:.3f}s ({tree_time/total_time*100:.1f}%)")
        print(f"ICP iterations:          {actual_iterations}")
        print(f"Average iter time:       {np.mean(iteration_times):.3f}s")
        print(f"Average correspondence:  {np.mean(correspondence_times):.3f}s")
        print(f"Average transform comp:  {np.mean(transform_times):.3f}s")
        print(f"Points per second:       {len(source_points)*actual_iterations/total_time:,.0f}")
        print(f"Initial distance:        {mean_distances[0]:.4f}")
        print(f"Final distance:          {mean_distances[-1]:.4f}")
        print(f"Improvement:             {mean_distances[0] - mean_distances[-1]:.4f} "
              f"({(1 - mean_distances[-1]/mean_distances[0])*100:.1f}%)")
        print(f"{'='*70}\n")
        
        # If mirroring was applied, combine it with the transformation
        if mirroring_matrix is not None:
            cumulative_transform = cumulative_transform @ mirroring_matrix
        
        return cumulative_transform, mean_distances, intermediate_transforms
    
    def register_with_mirroring(self, max_iterations=100, visualize=False):

        total_start = time.time()
        
        print("\n" + "="*80)
        print("ICP with Automatic Mirroring Detection")
        print("="*80)
        print("Testing all 8 possible mirroring combinations...\n")
        
        mirroring_options = generate_mirroring_matrices()
        results = []
        
        # Save original source points
        original_low_res = self.source.low_res.copy()
        
        # Test each mirroring configuration
        for idx, (description, mirroring_matrix) in enumerate(mirroring_options, 1):
            print(f"\n[{idx}/8] Testing: {description}")
            print("-" * 60)
            
            # Create mirrored source
            mirrored_source = PointCloud(self.source.to_o3d())
            mirrored_source._low_res = apply_mirroring_transform(original_low_res, mirroring_matrix)
            
            # Create temporary ICP instance
            temp_icp = ICPRegistration(mirrored_source, self.target)
            temp_icp.initial_transform = np.eye(4)
            
            try:
                transform, distances, _ = temp_icp.register(
                    max_iterations=max_iterations,
                    quality="low",
                    visualize=False
                )
                
                final_distance = distances[-1]
                print(f"  Final distance: {final_distance:.4f}")
                
                results.append({
                    'description': description,
                    'mirroring_matrix': mirroring_matrix,
                    'transformation': transform,
                    'distances': distances,
                    'final_distance': final_distance
                })
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not results:
            raise RuntimeError("All mirroring attempts failed!")
        
        # Select best result
        best_result = min(results, key=lambda x: x['final_distance'])
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        results_sorted = sorted(results, key=lambda x: x['final_distance'])
        for idx, result in enumerate(results_sorted, 1):
            marker = " *** BEST ***" if result == best_result else ""
            print(f"{idx}. {result['description']:30s} - {result['final_distance']:.6f}{marker}")
        
        print(f"\nBest configuration: {best_result['description']}")
        print(f"Best distance: {best_result['final_distance']:.6f}")
        print("="*80 + "\n")
        
        # Refinement phase at medium resolution
        refinement_start = time.time()
        print("\n" + "="*80)
        print("REFINEMENT PHASE")
        print("="*80)
        print(f"Starting from distance: {best_result['final_distance']:.6f}")
        print("Refining at medium resolution...\n")
        
        # Apply best mirroring to source
        refined_source = PointCloud(self.source.to_o3d())
        refined_source._medium_res = apply_mirroring_transform(
            self.source.medium_res, best_result['mirroring_matrix']
        )
        
        # Create refined ICP instance
        refined_icp = ICPRegistration(refined_source, self.target)
        refined_icp.initial_transform = best_result['transformation'].copy()
        
        refined_transform, refined_distances, refined_intermediates = refined_icp.register(
            max_iterations=100,
            quality="medium",
            target_distance=9.0,
            visualize=visualize
        )
        
        refinement_time = time.time() - refinement_start
        total_time = time.time() - total_start
        
        print(f"\nRefinement complete!")
        print(f"Final distance: {refined_distances[-1]:.6f}")
        print(f"Refinement iterations: {len(refined_distances)}")
        
        # Print overall timing summary
        print(f"\n{'='*80}")
        print(f"OVERALL MIRRORING DETECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total runtime:              {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"  Initial search (8 configs): {refinement_start - total_start:.2f}s")
        print(f"  Refinement phase:          {refinement_time:.2f}s")
        print(f"Best configuration:         {best_result['description']}")
        print(f"Initial distance:           {best_result['final_distance']:.6f}")
        print(f"Final distance:             {refined_distances[-1]:.6f}")
        print(f"Total improvement:          {best_result['final_distance'] - refined_distances[-1]:.6f}")
        print(f"{'='*80}\n")
        
        # Combine transformations
        combined_transform = refined_transform @ best_result['mirroring_matrix']
        all_distances = best_result['distances'] + refined_distances
        
        # Build mirroring info
        mirroring_info = {
            'description': best_result['description'],
            'mirroring_matrix': best_result['mirroring_matrix'],
            'initial_distance': best_result['final_distance'],
            'final_distance': refined_distances[-1],
            'refinement_iterations': len(refined_distances),
            'all_results': results_sorted
        }
        
        return combined_transform, all_distances, refined_intermediates, mirroring_info
    
    def visualize_initial(self):
        """
        Visualize the initial state before alignment (full quality with colors).
        """
        # Use full resolution with original colors
        source_pcd = self.source.to_o3d()
        target_pcd = self.target.to_o3d()
        
        # If no colors, use default coloring: red for source, blue for target
        if self.source.colors is None:
            source_pcd.paint_uniform_color([1, 0, 0])
        if self.target.colors is None:
            target_pcd.paint_uniform_color([0, 0, 1])
        
        o3d.visualization.draw_geometries(
            [source_pcd, target_pcd],
            window_name="Initial State (Before ICP)",
            width=1024,
            height=768
        )
    
    def visualize_final(self, transformation, mirroring_info=None):
        """
        Visualize the final aligned state (full quality with colors).
        
        Args:
            transformation: 4x4 transformation matrix
            mirroring_info: Optional mirroring information dictionary
        """
        # Apply transformation to source points (full quality)
        source_transformed = apply_transformation(
            self.source.points, transformation
        )
        
        # Create Open3D point clouds with colors
        source_pcd = self.source.to_o3d(points=source_transformed)
        target_pcd = self.target.to_o3d()
        
        # If no colors, use default coloring
        if self.source.colors is None:
            source_pcd.paint_uniform_color([1, 0, 0])
        if self.target.colors is None:
            target_pcd.paint_uniform_color([0, 0, 1])
        
        # Window title
        window_name = "Final State (After ICP Alignment)"
        if mirroring_info:
            desc = mirroring_info['description']
            dist = mirroring_info['final_distance']
            window_name += f" - {desc} (distance: {dist:.4f})"
        
        o3d.visualization.draw_geometries(
            [source_pcd, target_pcd],
            window_name=window_name,
            width=1024,
            height=768
        )
    
    def save_result(self, filepath, transformation, mean_distances, 
                    intermediate_transforms, mirroring_info=None):
        """Save registration results to file."""
        result = {
            'transformation': transformation,
            'mean_distances': mean_distances,
            'intermediate_transforms': intermediate_transforms,
            'source_points': self.source.points,
            'target_points': self.target.points,
            'mirroring_info': mirroring_info
        }
        
        if self.source.colors is not None:
            result['source_colors'] = self.source.colors
        if self.target.colors is not None:
            result['target_colors'] = self.target.colors
        
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def load_result(filepath):
        """Load previously saved registration results."""
        if not os.path.exists(filepath):
            print(f"File {filepath} not found")
            return None
        
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        print(f"Results loaded from {filepath}")
        return result
    
    def _get_points(self, point_cloud, quality):
        """Get points at specified quality level."""
        if quality == "full":
            return point_cloud.points
        elif quality == "medium":
            return point_cloud.medium_res
        elif quality == "low":
            return point_cloud.low_res
        else:
            raise ValueError(f"Unknown quality: {quality}")
    
    def _find_correspondences(self, source_points, tree_root, target_points, n_jobs=4):
        """Find nearest neighbor correspondences using parallel processing."""
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(nearest_neighbor_search)(p, tree_root, target_points) 
            for p in source_points
        )
        neighbors, distances = zip(*results)
        return np.array(neighbors), np.array(distances)
    
    def _setup_visualization(self, source_points, target_points, loss_fn):
        """Setup Open3D visualization window."""
        vis = o3d.visualization.Visualizer()
        window_title = f"ICP Progress (Loss: {loss_fn})"
        vis.create_window(window_name=window_title, width=1024, height=768)
        
        source_vis = o3d.geometry.PointCloud()
        target_vis = o3d.geometry.PointCloud()
        
        source_vis.points = o3d.utility.Vector3dVector(source_points)
        target_vis.points = o3d.utility.Vector3dVector(target_points)
        
        source_vis.paint_uniform_color([1, 0, 0])  # Red
        target_vis.paint_uniform_color([0, 0, 1])  # Blue
        
        vis.add_geometry(source_vis)
        vis.add_geometry(target_vis)
        
        return vis, source_vis, target_vis
    
    def _get_mirroring_matrix(self, mirror_axes):
        """
        Get mirroring transformation matrix for specified axes.
        
        Args:
            mirror_axes: String specifying axes to mirror ('X', 'Y', 'Z', 'XY', etc.)
        
        Returns:
            4x4 mirroring transformation matrix
        """
        matrix = np.eye(4)
        
        mirror_axes = mirror_axes.upper()
        if 'X' in mirror_axes:
            matrix[0, 0] = -1
        if 'Y' in mirror_axes:
            matrix[1, 1] = -1
        if 'Z' in mirror_axes:
            matrix[2, 2] = -1
        
        return matrix

