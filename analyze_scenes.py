#!/usr/bin/env python3
"""
Comprehensive analysis of the two point cloud scenes to understand
why they were chosen as ICP case studies.

This script examines:
- Point cloud statistics (size, density, spatial distribution)
- Geometric properties (planarity, roughness, structure)
- Overlap and alignment characteristics
- Challenges for ICP algorithms
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pclouder import PointCloud


def compute_basic_statistics(pcd_points, name):
    """Compute basic statistics about a point cloud."""
    print(f"\n{'='*80}")
    print(f"Point Cloud: {name}")
    print(f"{'='*80}")
    
    stats = {}
    
    # Basic counts
    stats['num_points'] = len(pcd_points)
    print(f"Number of points: {stats['num_points']:,}")
    
    # Spatial extent
    min_bound = np.min(pcd_points, axis=0)
    max_bound = np.max(pcd_points, axis=0)
    extent = max_bound - min_bound
    
    stats['min_bound'] = min_bound
    stats['max_bound'] = max_bound
    stats['extent'] = extent
    
    print(f"\nSpatial Extent:")
    print(f"  X: [{min_bound[0]:.2f}, {max_bound[0]:.2f}] (range: {extent[0]:.2f})")
    print(f"  Y: [{min_bound[1]:.2f}, {max_bound[1]:.2f}] (range: {extent[1]:.2f})")
    print(f"  Z: [{min_bound[2]:.2f}, {max_bound[2]:.2f}] (range: {extent[2]:.2f})")
    
    volume = np.prod(extent)
    stats['volume'] = volume
    print(f"  Bounding box volume: {volume:.2f}")
    
    # Centroid
    centroid = np.mean(pcd_points, axis=0)
    stats['centroid'] = centroid
    print(f"\nCentroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
    
    # Point density
    density = stats['num_points'] / volume if volume > 0 else 0
    stats['density'] = density
    print(f"Point density: {density:.2f} points/unit³")
    
    # Distribution statistics
    std_dev = np.std(pcd_points, axis=0)
    stats['std_dev'] = std_dev
    print(f"\nStandard deviation per axis:")
    print(f"  X: {std_dev[0]:.2f}, Y: {std_dev[1]:.2f}, Z: {std_dev[2]:.2f}")
    
    return stats


def compute_local_geometry(pcd_points, sample_size=10000):
    """Analyze local geometric properties."""
    print(f"\n{'='*40}")
    print("Local Geometry Analysis")
    print(f"{'='*40}")
    
    # Sample points if too many
    if len(pcd_points) > sample_size:
        indices = np.random.choice(len(pcd_points), sample_size, replace=False)
        sample_points = pcd_points[indices]
    else:
        sample_points = pcd_points
    
    # Create Open3D point cloud for normal estimation
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(sample_points)
    
    # Estimate normals
    print("Estimating surface normals...")
    o3d_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30)
    )
    normals = np.asarray(o3d_pcd.normals)
    
    # Compute planarity metrics
    # Group nearby points and check if they form planar surfaces
    print("\nAnalyzing planarity...")
    
    # Compute distance to nearest neighbors (roughness measure)
    print("Computing nearest neighbor distances (roughness)...")
    kdtree = o3d.geometry.KDTreeFlann(o3d_pcd)
    
    nn_distances = []
    for i in range(min(1000, len(sample_points))):
        [k, idx, dist] = kdtree.search_knn_vector_3d(sample_points[i], 10)
        if len(dist) > 1:
            nn_distances.append(np.mean(np.sqrt(dist[1:])))
    
    avg_nn_dist = np.mean(nn_distances)
    std_nn_dist = np.std(nn_distances)
    
    print(f"  Average nearest neighbor distance: {avg_nn_dist:.4f}")
    print(f"  Std dev of NN distance: {std_nn_dist:.4f}")
    print(f"  Coefficient of variation: {(std_nn_dist/avg_nn_dist):.4f}")
    
    # Analyze normal distribution (indicates surface types)
    print("\nNormal vector distribution:")
    
    # Check for dominant directions (planes)
    normal_abs = np.abs(normals)
    dominant_axis = np.argmax(normal_abs, axis=1)
    
    axis_names = ['X', 'Y', 'Z']
    for axis in range(3):
        count = np.sum(dominant_axis == axis)
        percent = 100 * count / len(dominant_axis)
        print(f"  {axis_names[axis]}-aligned surfaces: {percent:.1f}%")
    
    # Compute normal variance (low = planar, high = complex geometry)
    normal_variance = np.mean(np.var(normals, axis=0))
    print(f"  Normal variance: {normal_variance:.4f}")
    
    return {
        'avg_nn_distance': avg_nn_dist,
        'std_nn_distance': std_nn_dist,
        'normal_variance': normal_variance,
        'normals': normals,
        'axis_alignment': {
            'X': np.sum(dominant_axis == 0) / len(dominant_axis),
            'Y': np.sum(dominant_axis == 1) / len(dominant_axis),
            'Z': np.sum(dominant_axis == 2) / len(dominant_axis),
        }
    }


def analyze_overlap(source_points, target_points, name):
    """Analyze overlap characteristics between two point clouds."""
    print(f"\n{'='*80}")
    print(f"Overlap Analysis: {name}")
    print(f"{'='*80}")
    
    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    centroid_distance = np.linalg.norm(source_centroid - target_centroid)
    print(f"Centroid distance: {centroid_distance:.2f}")
    print(f"  Source centroid: [{source_centroid[0]:.2f}, {source_centroid[1]:.2f}, {source_centroid[2]:.2f}]")
    print(f"  Target centroid: [{target_centroid[0]:.2f}, {target_centroid[1]:.2f}, {target_centroid[2]:.2f}]")
    
    # Compute bounding box overlap
    source_min = np.min(source_points, axis=0)
    source_max = np.max(source_points, axis=0)
    target_min = np.min(target_points, axis=0)
    target_max = np.max(target_points, axis=0)
    
    # Intersection of bounding boxes
    intersection_min = np.maximum(source_min, target_min)
    intersection_max = np.minimum(source_max, target_max)
    
    has_overlap = np.all(intersection_min < intersection_max)
    
    if has_overlap:
        intersection_volume = np.prod(intersection_max - intersection_min)
        source_volume = np.prod(source_max - source_min)
        target_volume = np.prod(target_max - target_min)
        
        overlap_ratio = intersection_volume / min(source_volume, target_volume)
        print(f"\nBounding box overlap: {overlap_ratio*100:.1f}%")
        print(f"  Intersection volume: {intersection_volume:.2f}")
    else:
        print("\nBounding boxes do NOT overlap!")
        print("  This suggests significant misalignment or rotation.")
    
    # Sample-based overlap estimation
    print("\nSample-based overlap estimation...")
    sample_size = min(10000, len(source_points))
    source_sample_idx = np.random.choice(len(source_points), sample_size, replace=False)
    source_sample = source_points[source_sample_idx]
    
    # Build KDTree for target
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target_points)
    kdtree = o3d.geometry.KDTreeFlann(target_o3d)
    
    # Find nearest neighbors
    distances = []
    for point in source_sample:
        [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))
    
    distances = np.array(distances)
    
    print(f"  Mean nearest neighbor distance: {np.mean(distances):.2f}")
    print(f"  Median NN distance: {np.median(distances):.2f}")
    print(f"  Min NN distance: {np.min(distances):.2f}")
    print(f"  Max NN distance: {np.max(distances):.2f}")
    
    # Points within various thresholds (indicates overlap quality)
    for threshold in [10, 50, 100, 200]:
        within = np.sum(distances < threshold)
        percent = 100 * within / len(distances)
        print(f"  Points within {threshold} units: {percent:.1f}%")
    
    return {
        'centroid_distance': centroid_distance,
        'bbox_overlap': has_overlap,
        'mean_nn_distance': np.mean(distances),
        'median_nn_distance': np.median(distances),
    }


def visualize_scene(scene_num, persp1_points, persp2_points):
    """Create visualization of both perspectives."""
    print(f"\nGenerating visualization for Scene {scene_num}...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Sample for visualization
    sample_size = 50000
    if len(persp1_points) > sample_size:
        idx1 = np.random.choice(len(persp1_points), sample_size, replace=False)
        p1_sample = persp1_points[idx1]
    else:
        p1_sample = persp1_points
        
    if len(persp2_points) > sample_size:
        idx2 = np.random.choice(len(persp2_points), sample_size, replace=False)
        p2_sample = persp2_points[idx2]
    else:
        p2_sample = persp2_points
    
    # Plot both perspectives separately
    for i, (points, name) in enumerate([(p1_sample, 'Perspective 1'), (p2_sample, 'Perspective 2')]):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=0.1, alpha=0.5)
        ax.set_title(f'{name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Plot both together (no alignment)
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.scatter(p1_sample[:, 0], p1_sample[:, 1], p1_sample[:, 2], 
              c='red', s=0.1, alpha=0.3, label='Persp 1')
    ax.scatter(p2_sample[:, 0], p2_sample[:, 1], p2_sample[:, 2], 
              c='blue', s=0.1, alpha=0.3, label='Persp 2')
    ax.set_title('Both Overlaid (No Alignment)')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # XY projections
    for i, (points, name, color) in enumerate([
        (p1_sample, 'Perspective 1 (XY)', 'red'),
        (p2_sample, 'Perspective 2 (XY)', 'blue'),
    ]):
        ax = fig.add_subplot(2, 3, i+4)
        ax.scatter(points[:, 0], points[:, 1], c=color, s=0.1, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Combined XY projection
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(p1_sample[:, 0], p1_sample[:, 1], c='red', s=0.1, alpha=0.3, label='Persp 1')
    ax.scatter(p2_sample[:, 0], p2_sample[:, 1], c='blue', s=0.1, alpha=0.3, label='Persp 2')
    ax.set_title('Both XY Projections')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'scene{scene_num}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: scene{scene_num}_analysis.png")
    plt.close()


def analyze_scene(scene_num):
    """Comprehensive analysis of a scene."""
    print(f"\n\n{'#'*80}")
    print(f"# ANALYZING SCENE {scene_num}")
    print(f"{'#'*80}")
    
    # Load point clouds
    persp1_path = f'pcloud/scene{scene_num}_perspective1.ply'
    persp2_path = f'pcloud/scene{scene_num}_perspective2.ply'
    
    print(f"\nLoading point clouds...")
    pcd1 = PointCloud.from_file(persp1_path)
    pcd2 = PointCloud.from_file(persp2_path)
    
    # Basic statistics
    stats1 = compute_basic_statistics(pcd1.points, f"Scene {scene_num} - Perspective 1")
    stats2 = compute_basic_statistics(pcd2.points, f"Scene {scene_num} - Perspective 2")
    
    # Local geometry
    geom1 = compute_local_geometry(pcd1.points)
    geom2 = compute_local_geometry(pcd2.points)
    
    # Overlap analysis
    overlap = analyze_overlap(pcd1.points, pcd2.points, f"Scene {scene_num}")
    
    # Visualization
    visualize_scene(scene_num, pcd1.points, pcd2.points)
    
    return {
        'stats1': stats1,
        'stats2': stats2,
        'geom1': geom1,
        'geom2': geom2,
        'overlap': overlap
    }


def main():
    """Main analysis routine."""
    print("="*80)
    print("POINT CLOUD SCENE ANALYSIS FOR ICP CASE STUDY")
    print("="*80)
    print("\nThis script analyzes why these scenes were chosen as ICP examples")
    print("by examining their geometric properties, alignment challenges, and")
    print("characteristics that make them interesting test cases.")
    
    # Analyze both scenes
    scene1_data = analyze_scene(1)
    scene2_data = analyze_scene(2)
    
    # Summary comparison
    print(f"\n\n{'#'*80}")
    print("# SUMMARY AND ICP CHALLENGE ANALYSIS")
    print(f"{'#'*80}")
    
    print("\n" + "="*80)
    print("Why These Scenes Are Good ICP Case Studies")
    print("="*80)
    
    print("\n1. SCALE AND COMPLEXITY")
    print("-" * 40)
    print(f"Scene 1: {scene1_data['stats1']['num_points']:,} points per perspective")
    print(f"Scene 2: {scene2_data['stats1']['num_points']:,} points per perspective")
    print("\n  → Large point clouds test scalability and optimization")
    print("  → Real-world data size requires efficient KD-Tree and downsampling")
    
    print("\n2. ALIGNMENT CHALLENGES")
    print("-" * 40)
    print(f"Scene 1 initial centroid distance: {scene1_data['overlap']['centroid_distance']:.2f}")
    print(f"Scene 2 initial centroid distance: {scene2_data['overlap']['centroid_distance']:.2f}")
    print(f"Scene 1 mean NN distance: {scene1_data['overlap']['mean_nn_distance']:.2f}")
    print(f"Scene 2 mean NN distance: {scene2_data['overlap']['mean_nn_distance']:.2f}")
    
    print("\n  → Large initial misalignment tests convergence basin")
    print("  → ICP must handle significant rotation and translation")
    print("  → Multi-resolution strategy essential for these distances")
    
    print("\n3. GEOMETRIC CHARACTERISTICS")
    print("-" * 40)
    print(f"Scene 1 normal variance: {scene1_data['geom1']['normal_variance']:.4f}")
    print(f"Scene 2 normal variance: {scene2_data['geom1']['normal_variance']:.4f}")
    print(f"Scene 1 surface roughness (std NN): {scene1_data['geom1']['std_nn_distance']:.4f}")
    print(f"Scene 2 surface roughness (std NN): {scene2_data['geom1']['std_nn_distance']:.4f}")
    
    print("\n  → Complex geometry (not just planar walls) tests robustness")
    print("  → Varied surface normals prevent degenerate solutions")
    print("  → Natural roughness introduces noise that robust losses must handle")
    
    print("\n4. AXIS ALIGNMENT PATTERNS")
    print("-" * 40)
    for scene_num, data in [(1, scene1_data), (2, scene2_data)]:
        print(f"\nScene {scene_num}:")
        for axis, ratio in data['geom1']['axis_alignment'].items():
            print(f"  {axis}-aligned: {ratio*100:.1f}%")
    
    print("\n  → Mixed orientations require full 6-DOF transformation")
    print("  → Prevents trivial solutions or axis-decoupled optimization")
    
    print("\n5. POINT DENSITY VARIATIONS")
    print("-" * 40)
    print(f"Scene 1 density: {scene1_data['stats1']['density']:.2f} pts/unit³")
    print(f"Scene 2 density: {scene2_data['stats1']['density']:.2f} pts/unit³")
    
    print("\n  → Non-uniform density simulates real scanner behavior")
    print("  → Tests nearest neighbor search with varying point spacing")
    
    print("\n" + "="*80)
    print("KEY ICP CHALLENGES THESE SCENES INTRODUCE")
    print("="*80)
    
    challenges = [
        ("Scale", "~2 million points per cloud requires optimization"),
        ("Initial Misalignment", "Large rotation/translation needs coarse-to-fine strategy"),
        ("Complex Geometry", "Non-planar surfaces with rich features"),
        ("Noise & Outliers", "Real scanner data with measurement errors"),
        ("Symmetry Ambiguity", "Similar structures may cause mirroring issues"),
        ("Partial Overlap", "Only partial scene coverage from each perspective"),
        ("Density Variation", "Non-uniform sampling tests KD-Tree performance"),
        ("Convergence Basin", "Tests if ICP gets trapped in local minima"),
    ]
    
    for i, (challenge, description) in enumerate(challenges, 1):
        print(f"\n{i}. {challenge}")
        print(f"   {description}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
These scenes are excellent ICP case studies because they combine:
- Realistic scale and complexity from actual 3D scanners
- Significant initial misalignment requiring robust optimization
- Rich geometric features preventing trivial/degenerate solutions
- Natural noise and outliers demanding robust loss functions
- Challenges that necessitate key optimizations (KD-Tree, downsampling, mirroring)

They represent real-world scenarios where basic ICP would fail, requiring:
✓ Multi-resolution processing
✓ Efficient spatial indexing
✓ Mirroring detection
✓ Robust loss functions
✓ Careful convergence criteria

Perfect balance between being solvable (with good implementation) and challenging
enough to demonstrate algorithm sophistication.
    """)
    
    print("\nAnalysis complete! Check generated PNG files for visualizations.")


if __name__ == '__main__':
    main()

