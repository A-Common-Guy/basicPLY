#!/usr/bin/env python3
"""
Main entry point for ICP point cloud registration.

This script provides a clean command-line interface for running various
ICP registration modes.
"""

import argparse
import sys
from pathlib import Path

from pclouder import ICPRegistration, PointCloud
from pclouder.visualization import plot_convergence, plot_loss_comparison


def run_standard_icp(source_path, target_path, visualize=False, loss_fn='none', 
                     loss_params=None, save_results=True, apply_mirroring=None,
                     method='point_to_point'):
    """Run standard ICP registration."""
    print("\n" + "="*80)
    print("Standard ICP Registration")
    print("="*80)
    
    # Load point clouds
    print(f"\nLoading point clouds...")
    print(f"  Source: {source_path}")
    print(f"  Target: {target_path}")
    
    icp = ICPRegistration(source_path, target_path)
    
    print(f"  Source points: {len(icp.source)}")
    print(f"  Target points: {len(icp.target)}")
    
    # Run ICP
    mirroring_info = f" with {apply_mirroring}-axis mirroring" if apply_mirroring else ""
    print(f"\nRunning ICP (method={method}, loss={loss_fn}, visualize={visualize}{mirroring_info})...")
    transformation, mean_distances, intermediates = icp.register(
        max_iterations=100,
        quality="medium",
        loss_fn=loss_fn,
        loss_params=loss_params,
        visualize=visualize,
        apply_mirroring=apply_mirroring,
        method=method
    )
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)
    print(f"Final mean distance: {mean_distances[-1]:.4f}")
    print(f"Iterations: {len(mean_distances)}")
    print(f"\nTransformation matrix:")
    print(transformation)
    
    # Save results
    if save_results:
        icp.save_result('icp_results.pkl', transformation, mean_distances, 
                       intermediates)
    
    # Visualize initial and final states
    print("\nShowing initial state (before alignment)...")
    icp.visualize_initial()
    
    print("\nShowing final state (after alignment)...")
    icp.visualize_final(transformation)
    
    return transformation, mean_distances


def run_mirroring_icp(source_path, target_path, visualize=False, save_results=True):
    """Run ICP with automatic mirroring detection."""
    print("\n" + "="*80)
    print("ICP with Automatic Mirroring Detection")
    print("="*80)
    
    # Load point clouds
    print(f"\nLoading point clouds...")
    print(f"  Source: {source_path}")
    print(f"  Target: {target_path}")
    
    icp = ICPRegistration(source_path, target_path)
    
    print(f"  Source points: {len(icp.source)}")
    print(f"  Target points: {len(icp.target)}")
    
    # Run ICP with mirroring
    transformation, mean_distances, intermediates, mirroring_info = \
        icp.register_with_mirroring(max_iterations=100, visualize=visualize)
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best configuration: {mirroring_info['description']}")
    print(f"Final mean distance: {mean_distances[-1]:.4f}")
    print(f"Total iterations: {len(mean_distances)}")
    print(f"\nAll tested configurations:")
    for idx, res in enumerate(mirroring_info['all_results'], 1):
        marker = " (SELECTED)" if res['description'] == mirroring_info['description'] else ""
        print(f"  {idx}. {res['description']:30s} - {res['final_distance']:.6f}{marker}")
    
    # Save results
    if save_results:
        icp.save_result('icp_results.pkl', transformation, mean_distances, 
                       intermediates, mirroring_info)
    
    # Visualize initial and final states
    print("\nShowing initial state (before alignment)...")
    icp.visualize_initial()
    
    print("\nShowing final state (after alignment)...")
    icp.visualize_final(transformation, mirroring_info)
    
    return transformation, mean_distances, mirroring_info


def compare_loss_functions(source_path, target_path):
    """Compare different robust loss functions."""
    print("\n" + "="*80)
    print("Comparing Robust Loss Functions")
    print("="*80)
    
    loss_configs = [
        ('none', None, 'Standard Least Squares'),
        ('huber', {'delta': 10.0}, 'Huber Loss (delta=10.0)'),
        ('tukey', {'c': 15.0}, 'Tukey Biweight (c=15.0)'),
        ('percentile', {'percentile': 85}, 'Percentile Filter (85%)'),
    ]
    
    results = []
    
    for loss_fn, loss_params, description in loss_configs:
        print(f"\n{'='*80}")
        print(f"Testing: {description}")
        print("="*80)
        
        icp = ICPRegistration(source_path, target_path)
        
        transformation, mean_distances, _ = icp.register(
            max_iterations=100,
            quality="medium",
            loss_fn=loss_fn,
            loss_params=loss_params,
            visualize=False
        )
        
        results.append({
            'description': description,
            'loss_fn': loss_fn,
            'final_distance': mean_distances[-1],
            'iterations': len(mean_distances),
            'mean_distances': mean_distances,
        })
        
        print(f"\nResults: distance={mean_distances[-1]:.4f}, iterations={len(mean_distances)}")
    
    # Plot comparison
    plot_loss_comparison(results)


def load_and_visualize(filepath='icp_results.pkl'):
    """Load and visualize previously saved results."""
    print("\n" + "="*80)
    print("Loading Saved Results")
    print("="*80)
    
    result = ICPRegistration.load_result(filepath)
    if result is None:
        return
    
    mirroring_info = result.get('mirroring_info', None)
    
    print(f"\nLoaded results:")
    print(f"  Final distance: {result['mean_distances'][-1]:.4f}")
    print(f"  Iterations: {len(result['mean_distances'])}")
    
    if mirroring_info:
        print(f"  Configuration: {mirroring_info['description']}")
        print(f"  Improvement: {mirroring_info['initial_distance'] - mirroring_info['final_distance']:.4f}")
    
    # Reconstruct point clouds
    source = PointCloud(PointCloud.from_file('pcloud/scene1_perspective1.ply').to_o3d())
    target = PointCloud(PointCloud.from_file('pcloud/scene1_perspective2.ply').to_o3d())
    
    icp = ICPRegistration(source, target)
    
    # Visualize initial and final states
    print("\nShowing initial state...")
    icp.visualize_initial()
    
    print("\nShowing final state...")
    icp.visualize_final(result['transformation'], mirroring_info)


def main():
    parser = argparse.ArgumentParser(
        description='ICP Point Cloud Registration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ICP
  python run_icp.py standard pcloud/scene1_perspective1.ply pcloud/scene1_perspective2.ply
  
  # ICP with automatic mirroring detection
  python run_icp.py mirroring pcloud/scene1_perspective1.ply pcloud/scene1_perspective2.ply
  
  # ICP with visualization
  python run_icp.py standard pcloud/scene1_perspective1.ply pcloud/scene1_perspective2.ply --visualize
  
  # ICP with robust loss function
  python run_icp.py standard pcloud/scene1_perspective1.ply pcloud/scene1_perspective2.ply --loss huber
  
  # Compare loss functions
  python run_icp.py compare pcloud/scene1_perspective1.ply pcloud/scene1_perspective2.ply
  
  # Load and visualize saved results
  python run_icp.py load
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Registration mode')
    
    # Standard ICP
    standard_parser = subparsers.add_parser('standard', help='Standard ICP registration')
    standard_parser.add_argument('--source', type=str, help='Path to source point cloud', default=None)
    standard_parser.add_argument('--target', type=str, help='Path to target point cloud', default=None)
    standard_parser.add_argument('--visualize', action='store_true', 
                                help='Show real-time visualization')
    standard_parser.add_argument('--loss', type=str, default='none',
                                choices=['none', 'huber', 'tukey', 'percentile'],
                                help='Robust loss function')
    standard_parser.add_argument('--mirror', type=str, default=None,
                                help='Apply mirroring to source before ICP (e.g., X, Y, Z, XY, XZ, YZ, XYZ)')
    standard_parser.add_argument('--method', type=str, default='point_to_point',
                                choices=['point_to_point', 'point_to_plane'],
                                help='ICP method: point_to_point (default) or point_to_plane (faster convergence)')
    standard_parser.add_argument('--no-save', action='store_true',
                                help='Do not save results')
    
    # Mirroring ICP
    mirror_parser = subparsers.add_parser('mirroring', 
                                          help='ICP with automatic mirroring detection')
    mirror_parser.add_argument('--source', type=str, help='Path to source point cloud', default=None)
    mirror_parser.add_argument('--target', type=str, help='Path to target point cloud', default=None)
    mirror_parser.add_argument('--visualize', action='store_true',
                              help='Show visualization of final result')
    mirror_parser.add_argument('--no-save', action='store_true',
                              help='Do not save results')
    
    # Compare loss functions
    compare_parser = subparsers.add_parser('compare', 
                                           help='Compare different loss functions')
    compare_parser.add_argument('--source', type=str, help='Path to source point cloud', default=None)
    compare_parser.add_argument('--target', type=str, help='Path to target point cloud', default=None)
    
    # Load results
    load_parser = subparsers.add_parser('load', help='Load and visualize saved results')
    load_parser.add_argument('--file', type=str, default='icp_results.pkl',
                            help='Path to saved results file')
    
    args = parser.parse_args()


    if args.source is None:
        args.source = 'pcloud/scene2_perspective1.ply'
    if args.target is None:
        args.target = 'pcloud/scene2_perspective2.ply'
    
    # Default to mirroring mode with default files if no arguments
    if args.mode is None:
        print("No mode specified. Running mirroring mode with default point clouds...")
        run_mirroring_icp(
            'pcloud/scene1_perspective1.ply',
            'pcloud/scene1_perspective2.ply',
            visualize=False,
            save_results=True
        )
        return
    
    # Execute requested mode
    if args.mode == 'standard':
        run_standard_icp(
            args.source, 
            args.target, 
            visualize=args.visualize,
            loss_fn=args.loss,
            save_results=not args.no_save,
            apply_mirroring=args.mirror,
            method=args.method
        )
    elif args.mode == 'mirroring':
        run_mirroring_icp(
            args.source,
            args.target,
            visualize=args.visualize,
            save_results=not args.no_save
        )
    elif args.mode == 'compare':
        compare_loss_functions(args.source, args.target)
    elif args.mode == 'load':
        load_and_visualize(args.file)


if __name__ == '__main__':
    main()

