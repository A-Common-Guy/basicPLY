"""Visualization utilities for ICP results."""

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(mean_distances, mirroring_info=None, save_path='icp_convergence.png'):
    """
    Plot ICP convergence curve.
    
    Args:
        mean_distances: List of mean distances over iterations
        mirroring_info: Optional dictionary with mirroring information
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot convergence curve
    ax.plot(mean_distances, marker='o', linewidth=2, markersize=4, 
            color='#2E86AB', label='Mean Distance')
    
    # Add target distance line if using mirroring
    if mirroring_info and 'initial_distance' in mirroring_info:
        ax.axhline(y=9.0, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Target Distance (9.0)')
        
        # Mark refinement phase start
        if 'refinement_iterations' in mirroring_info:
            refinement_start = len(mean_distances) - mirroring_info['refinement_iterations']
            if refinement_start > 0:
                ax.axvline(x=refinement_start, color='green', linestyle='--', 
                          linewidth=1.5, alpha=0.5, label='Refinement Phase Start')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Distance', fontsize=12)
    
    # Build title
    title = 'ICP Convergence'
    if mirroring_info:
        if 'initial_distance' in mirroring_info:
            title += f"\n{mirroring_info['description']}"
            initial = mirroring_info['initial_distance']
            final = mirroring_info['final_distance']
            improvement = initial - final
            title += f"\nInitial: {initial:.4f} → Final: {final:.4f} (Δ: {improvement:.4f})"
        else:
            title += f"\n{mirroring_info['description']} (Final: {mirroring_info['final_distance']:.4f})"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Convergence plot saved to '{save_path}'")
    plt.show()


def plot_loss_comparison(results, save_path='loss_comparison.png'):
    """
    Compare different loss functions side-by-side.
    
    Args:
        results: List of dictionaries with 'description', 'mean_distances', 
                 'final_distance', 'iterations' keys
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot convergence curves
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    for result, color in zip(results, colors):
        ax1.plot(result['mean_distances'], label=result['description'], 
                color=color, linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Mean Distance', fontsize=12)
    ax1.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of final distances
    descriptions = [r['description'] for r in results]
    final_distances = [r['final_distance'] for r in results]
    iterations = [r['iterations'] for r in results]
    
    x_pos = np.arange(len(descriptions))
    ax2.bar(x_pos, final_distances, color=colors[:len(results)], alpha=0.7)
    ax2.set_xlabel('Loss Function', fontsize=12)
    ax2.set_ylabel('Final Mean Distance', fontsize=12)
    ax2.set_title('Final Distance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([d.split('(')[0].strip() for d in descriptions], 
                        rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (dist, iters) in enumerate(zip(final_distances, iterations)):
        ax2.text(i, dist, f'{dist:.2f}\n({iters} iter)', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nComparison plot saved to '{save_path}'")
    plt.show()
    
    # Print summary table
    print("\nFinal Results:")
    print(f"{'Loss Function':<35} {'Final Distance':<15} {'Iterations':<12}")
    print("-" * 62)
    for result in results:
        print(f"{result['description']:<35} {result['final_distance']:<15.4f} {result['iterations']:<12}")
    
    best = min(results, key=lambda x: x['final_distance'])
    print(f"\n✓ Best result: {best['description']} (distance: {best['final_distance']:.4f})")

