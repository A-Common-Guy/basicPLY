import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pclouder.pclouder import PClouder
from pclouder.Node import KDTree
import os


def run_icp_and_save(visualize=False, use_mirroring=False):
    """Run ICP and save the results."""
    print("Loading point clouds...")
    pcd1 = o3d.io.read_point_cloud("pcloud/scene2_perspective1.ply")
    pcd2 = o3d.io.read_point_cloud("pcloud/scene2_perspective2.ply")

    
    pclouder = PClouder(pcd1, pcd2)
    print(f"Point cloud 1 shape: {pclouder.p1_points_full.shape}")
    print(f"Point cloud 2 shape: {pclouder.p2_points_full.shape}")

    # Run ICP with optional mirroring detection
    mirroring_info = None
    if use_mirroring:
        print(f"\nRunning ICP with automatic mirroring detection (visualize={visualize})...")
        cumulative_transformation, mean_distances, intermediate_transformations, mirroring_info = pclouder.apply_icp_with_mirroring(
            max_iterations=100, 
            visualize=visualize,
            )
    else:
        print(f"\nRunning standard ICP (visualize={visualize})...")
        cumulative_transformation, mean_distances, intermediate_transformations = pclouder.apply_icp(
            max_iterations=100,     
            visualize=visualize,
            quality="medium",
        )
    
    print(f"\nCumulative transformation:\n{cumulative_transformation}")
    print(f"\nFinal mean distance: {mean_distances[-1]:.4f}")
    print(f"Number of iterations: {len(mean_distances)}")

    # Save the results
    save_file = "icp_results.pkl"
    pclouder.save_result(save_file, cumulative_transformation, mean_distances, intermediate_transformations, mirroring_info)
    
    # Visualize the final result
    print("\nShowing final alignment...")
    pclouder.visualize_result(cumulative_transformation, mirroring_info)
    
    # Plot convergence
    plot_convergence(mean_distances, mirroring_info)


def load_and_visualize():
    """Load previously saved results and visualize them."""
    save_file = "icp_results.pkl"
    
    if not os.path.exists(save_file):
        print(f"No saved results found at {save_file}")
        print("Run the ICP first with run_icp_and_save()")
        return
    
    # Load the saved results
    result = PClouder.load_result(save_file)
    
    if result is None:
        return
    
    mirroring_info = result.get('mirroring_info', None)
    
    print(f"\nLoaded results:")
    print(f"- Final mean distance: {result['mean_distances'][-1]:.4f}")
    print(f"- Number of iterations: {len(result['mean_distances'])}")
    print(f"- Transformation matrix:\n{result['cumulative_transformation']}")
    
    if mirroring_info:
        print(f"\nMirroring information:")
        print(f"- Configuration: {mirroring_info['description']}")
        
        if 'initial_distance' in mirroring_info:
            print(f"- Initial distance (after mirroring selection): {mirroring_info['initial_distance']:.4f}")
            print(f"- Final distance (after refinement): {mirroring_info['final_distance']:.4f}")
            print(f"- Refinement iterations: {mirroring_info.get('refinement_iterations', 'N/A')}")
            print(f"- Improvement: {mirroring_info['initial_distance'] - mirroring_info['final_distance']:.4f}")
        else:
            print(f"- Final distance: {mirroring_info['final_distance']:.4f}")
        
        print(f"\nAll tested configurations:")
        for idx, res in enumerate(mirroring_info['all_results'], 1):
            marker = " (SELECTED)" if res['description'] == mirroring_info['description'] else ""
            print(f"  {idx}. {res['description']:30s} - {res['final_distance']:.6f}{marker}")
    
    # Create a PClouder instance from loaded data
    pcd1_downsampled = o3d.geometry.PointCloud()
    pcd2_downsampled = o3d.geometry.PointCloud()
    pcd1_downsampled.points = o3d.utility.Vector3dVector(result['p1_points'])
    pcd2_downsampled.points = o3d.utility.Vector3dVector(result['p2_points'])
    
    if 'p1_colors' in result:
        pcd1_downsampled.colors = o3d.utility.Vector3dVector(result['p1_colors'])
    if 'p2_colors' in result:
        pcd2_downsampled.colors = o3d.utility.Vector3dVector(result['p2_colors'])

    pclouder = PClouder(pcd1_downsampled, pcd2_downsampled)
    
    # Visualize the result
    pclouder.visualize_result(result['cumulative_transformation'], mirroring_info)
    
    # Plot convergence
    plot_convergence(result['mean_distances'], mirroring_info)


def plot_convergence(mean_distances, mirroring_info=None):
    """Plot the convergence of ICP over iterations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the convergence curve
    ax.plot(mean_distances, marker='o', linewidth=2, markersize=4, color='#2E86AB', label='Mean Distance')
    
    # Add horizontal line for target distance if applicable
    if mirroring_info and 'initial_distance' in mirroring_info:
        ax.axhline(y=9.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target Distance (9.0)')
        
        # Calculate where refinement phase starts (assuming initial search used fewer iterations)
        if 'refinement_iterations' in mirroring_info:
            refinement_start = len(mean_distances) - mirroring_info['refinement_iterations']
            if refinement_start > 0:
                ax.axvline(x=refinement_start, color='green', linestyle='--', linewidth=1.5, 
                          alpha=0.5, label=f'Refinement Phase Start')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Distance', fontsize=12)
    
    title = 'ICP Convergence'
    if mirroring_info:
        if 'initial_distance' in mirroring_info:
            title += f"\n{mirroring_info['description']}"
            title += f"\nInitial: {mirroring_info['initial_distance']:.4f} → Final: {mirroring_info['final_distance']:.4f}"
            title += f" (Improvement: {mirroring_info['initial_distance'] - mirroring_info['final_distance']:.4f})"
        else:
            title += f"\n{mirroring_info['description']} (Final: {mirroring_info['final_distance']:.4f})"
    
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('icp_convergence.png', dpi=150)
    print("Convergence plot saved to 'icp_convergence.png'")
    plt.show()


def main():
    print("=" * 60)
    print("ICP Point Cloud Registration")
    print("=" * 60)
    print("\nOptions:")
    print("1. Run ICP and save results (no visualization)")
    print("2. Run ICP with real-time visualization")
    print("3. Run ICP with automatic mirroring detection (no visualization)")
    print("4. Run ICP with automatic mirroring detection (with visualization)")
    print("5. Load and visualize saved results")
    print("\nEdit the code to choose which option to run.\n")
    
    # Choose one of these options:
    
    # Option 1: Run ICP without visualization (faster)
    # run_icp_and_save(visualize=False, use_mirroring=False)
    
    # Option 2: Run ICP with real-time visualization (slower, shows iterations)
    # run_icp_and_save(visualize=True, use_mirroring=False)
    
    # Option 3: Run ICP with automatic mirroring detection (no visualization, faster)
    run_icp_and_save(visualize=False, use_mirroring=True)
    
    # Option 4: Run ICP with automatic mirroring detection + visualization of best result
    # run_icp_and_save(visualize=True, use_mirroring=True)
    
    # Option 5: Load and visualize previously saved results
    # load_and_visualize()


if __name__ == "__main__":
    main()
