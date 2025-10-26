import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pclouder.utils import time_function
from joblib import Parallel, delayed
from pclouder.utils import compute_transformation, apply_transformation, distance, _nearest_neighbor_search_iterative, voxelize_point_cloud, generate_mirroring_matrices, apply_mirroring
from pclouder.Node import KDTree
import pickle
import os




class PClouder:



    
    def __init__(self, perspective_1_ply, perspective_2_ply):
        self.p1 = perspective_1_ply
        self.p2 = perspective_2_ply
        #3 qualities, full for final visualization, medium for refining and low for initial search
        self.p1_points_full = np.asarray(self.p1.points)
        self.p2_points_full = np.asarray(self.p2.points)

        if self.p1.has_colors():
            self.p1_colors_full = np.asarray(self.p1.colors)
        else:
            self.p1_colors_full = None
        
        if self.p2.has_colors():
            self.p2_colors_full = np.asarray(self.p2.colors)
        else:
            self.p2_colors_full = None

        self.p1_points_medium = voxelize_point_cloud(self.p1_points_full, 10)
        self.p2_points_medium = voxelize_point_cloud(self.p2_points_full, 10)
        self.p1_points_low = voxelize_point_cloud(self.p1_points_full, 20)
        self.p2_points_low = voxelize_point_cloud(self.p2_points_full, 20)

        

        self.initial_guess_transform = np.eye(4)


    def set_initial_guess_transform(self, initial_guess_transform):
        self.initial_guess_transform = initial_guess_transform


    def show_side_by_side(self):
        o3d.visualization.draw_geometries([self.p1, self.p2])

    def save_pcloud(self, pcloud, pcloud_path):
        o3d.io.write_point_cloud(pcloud_path, pcloud)
    
    def save_result(self, filename, cumulative_transformation, mean_distances, intermediate_transformations, 
                    mirroring_info=None):
        """Save the ICP results to a file."""
        result = {
            'cumulative_transformation': cumulative_transformation,
            'mean_distances': mean_distances,
            'intermediate_transformations': intermediate_transformations,
            'p1_points': self.p1_points_full,
            'p2_points': self.p2_points_full,
            'mirroring_info': mirroring_info  # Store mirroring information
        }
        if self.p1_colors_full is not None:
            result['p1_colors'] = self.p1_colors_full
        if self.p2_colors_full is not None:
            result['p2_colors'] = self.p2_colors_full

        with open(filename, 'wb') as f:
            pickle.dump(result, f)
        print(f"Results saved to {filename}")
    
    @staticmethod
    def load_result(filename):
        """Load previously saved ICP results."""
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return None
        with open(filename, 'rb') as f:
            result = pickle.load(f)
        print(f"Results loaded from {filename}")
        return result
    
    def visualize_result(self, cumulative_transformation, mirroring_info=None):
        """Visualize the final aligned point clouds."""
        source_transformed = apply_transformation(self.p1_points_full.copy(), cumulative_transformation)
        
        # Create point clouds for visualization
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        
        source_pcd.points = o3d.utility.Vector3dVector(source_transformed)
        target_pcd.points = o3d.utility.Vector3dVector(self.p2_points_full.copy())
        
        # Color: source = red, target = blue
        if self.p1_colors_full is not None and self.p2_colors_full is not None:
            source_pcd.colors = o3d.utility.Vector3dVector(self.p1_colors_full)
            target_pcd.colors = o3d.utility.Vector3dVector(self.p2_colors_full)
        else:
            source_pcd.paint_uniform_color([1, 0, 0])  # Red
            target_pcd.paint_uniform_color([0, 0, 1])  # Blue
        
        # Create window name with mirroring info if available
        window_name = "Final ICP Result"
        if mirroring_info:
            window_name += f" - {mirroring_info['description']} (score: {mirroring_info['final_distance']:.4f})"
        
        o3d.visualization.draw_geometries([source_pcd, target_pcd],
                                         window_name=window_name,
                                         width=1024, height=768)


    def apply_icp(self, max_iterations=100, visualize=False, target_distance=None, quality="medium"):  
        if quality == "full":
            p1_points = self.p1_points_full
            p2_points = self.p2_points_full
        elif quality == "medium":
            p1_points = self.p1_points_medium
            p2_points = self.p2_points_medium
        elif quality == "low":
            p1_points = self.p1_points_low
            p2_points = self.p2_points_low
        
        tree_builder=KDTree(leaf_size=128, dimension=p2_points.shape[1])
        root=tree_builder.build_optimized(p2_points)
        mean_distances = []
        intermediate_transformations = [self.initial_guess_transform.copy()]
        cumulative_transformation = self.initial_guess_transform
        
        # Setup visualization if requested
        vis = None
        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="ICP Progress", width=1024, height=768)
            
            # Create copies for visualization
            source_vis = o3d.geometry.PointCloud()
            target_vis = o3d.geometry.PointCloud()
            
            source_vis.points = o3d.utility.Vector3dVector(p1_points)
            target_vis.points = o3d.utility.Vector3dVector(p2_points)
            
            # Color: source = red, target = blue
            source_vis.paint_uniform_color([1, 0, 0])  # Red
            target_vis.paint_uniform_color([0, 0, 1])  # Blue
            
            vis.add_geometry(source_vis)
            vis.add_geometry(target_vis)
        
        for i in range(max_iterations):
            source_transformed=apply_transformation(p1_points, cumulative_transformation)
            print(f"Source transformed shape: {source_transformed.shape}")
            nearest_neighbors, distances = self.parallel_batch_distances(source_transformed, root, p2_points)
            
            mean_distance = np.mean(distances)
            mean_distances.append(mean_distance)

            
            # Update visualization
            if visualize and vis is not None:
                source_vis.points = o3d.utility.Vector3dVector(source_transformed)
                vis.update_geometry(source_vis)
                vis.poll_events()
                vis.update_renderer()

            
            # Check if target distance is reached
            if target_distance is not None and mean_distance < target_distance:
                print(f"ICP reached target distance ({target_distance}) in {i+1} iterations")
                break

            
            transformation = compute_transformation(source_transformed, nearest_neighbors)
            cumulative_transformation = transformation @ cumulative_transformation
            intermediate_transformations.append(cumulative_transformation.copy())
            print(f"Iteration {i}: Mean distance = {mean_distance}")

            #if derivative of mean distance is less than 0.01
            if i > 0 and mean_distances[i-1] - mean_distances[i] < 0.01:
                print(f"ICP converged in {i} iterations")
                break


        
        if visualize and vis is not None:
            vis.destroy_window()

    

            
        return cumulative_transformation, mean_distances, intermediate_transformations

    
    def apply_icp_with_mirroring(self, max_iterations=100, visualize=False):
       
        print("\n" + "="*80)
        print("Running ICP with Automatic Mirroring Detection")
        print("="*80)
        print("Testing all 8 possible mirroring combinations...\n")
        
        mirroring_options = generate_mirroring_matrices()
        results = []
        
        # Save original points
        original_p1_points = self.p1_points_low.copy()
        
        for idx, (description, mirroring_matrix) in enumerate(mirroring_options, 1):
            print(f"\n[{idx}/8] Testing: {description}")
            print("-" * 60)
            
            # Apply mirroring to source point cloud
            self.p1_points_low = apply_mirroring(original_p1_points, mirroring_matrix)
            
            # Reset initial guess for each trial
            self.initial_guess_transform = np.eye(4)
            
            # Run ICP without visualization for trials
            try:
                cumulative_transformation, mean_distances, intermediate_transformations = self.apply_icp(
                    max_iterations=max_iterations,
                    visualize=False,  # Don't visualize trials
                    quality="low"
                )
                
                final_distance = mean_distances[-1]
                print(f"  Final distance: {final_distance:.4f}")
                
                results.append({
                    'description': description,
                    'mirroring_matrix': mirroring_matrix,
                    'cumulative_transformation': cumulative_transformation,
                    'mean_distances': mean_distances,
                    'intermediate_transformations': intermediate_transformations,
                    'final_distance': final_distance
                })
            except Exception as e:
                print(f"  Error during ICP: {e}")
                continue
        
        if not results:
            print("\nAll mirroring attempts failed!")
            return None, None, None, None
        
        # Select the best result (lowest final distance)
        best_result = min(results, key=lambda x: x['final_distance'])
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        # Print all results sorted by score
        results_sorted = sorted(results, key=lambda x: x['final_distance'])
        for idx, result in enumerate(results_sorted, 1):
            marker = " *** BEST ***" if result == best_result else ""
            print(f"{idx}. {result['description']:30s} - Final distance: {result['final_distance']:.6f}{marker}")
        
        print(f"\nBest configuration: {best_result['description']}")
        print(f"Best final distance: {best_result['final_distance']:.6f}")
        print("="*80 + "\n")
        
        # Apply the best mirroring to the source points
        self.p1_points_medium = apply_mirroring(self.p1_points_medium, best_result['mirroring_matrix'])
        
        # Refinement phase: Continue ICP until 100 iterations or distance < 9
        print("\n" + "="*80)
        print("REFINEMENT PHASE")
        print("="*80)
        print(f"Starting refinement from distance: {best_result['final_distance']:.6f}")
        print("Continuing until 100 iterations OR distance < 9.0\n")
        
        # Set initial transformation from best result
        self.initial_guess_transform = best_result['cumulative_transformation'].copy()
        
        # Run ICP with refinement criteria
        refined_transformation, refined_mean_distances, refined_intermediate_transformations = self.apply_icp(
            max_iterations=100,
            visualize=visualize,
            target_distance=9.0,  # Stop if we reach this distance
            quality="medium"
        )
        
        print(f"\nRefinement complete!")
        print(f"Final distance after refinement: {refined_mean_distances[-1]:.6f}")
        print(f"Total iterations qualityin refinement: {len(refined_mean_distances)}")
        
        # Combine mirroring and refined transformation
        combined_transformation = refined_transformation @ best_result['mirroring_matrix']
        
        # Combine mean distances from initial search and refinement
        all_mean_distances = best_result['mean_distances'] + refined_mean_distances
        
        # Store mirroring info with refinement data
        mirroring_info = {
            'description': best_result['description'],
            'mirroring_matrix': best_result['mirroring_matrix'],
            'final_distance': refined_mean_distances[-1],
            'initial_distance': best_result['final_distance'],
            'refinement_iterations': len(refined_mean_distances),
            'all_results': results_sorted
        }
        
        # Visualize if requested
        if visualize:
            print("\nShowing best alignment with visualization...")
            self.visualize_result(refined_transformation, mirroring_info)
        
        return combined_transformation, all_mean_distances, refined_intermediate_transformations, mirroring_info


        

    def nearest_neighbor_search(self, query_point, node=None, depth=0, best=(None, np.inf)):
        if node is None:
            node = self.root
            best=(None, np.inf) 
            return best
        
        if node.indices is not None:
            leaf_points = self.p1_points[node.indices]
            for point in leaf_points:
                dist = distance(query_point, point)
                if dist < best[1]:
                    best = (point, dist)
            return best

        if node.point is not None:
            dist = distance(query_point, node.point)
            if dist < best[1]:
                best = (node.point, dist)
        
        axis = node.axis
        #check if the query point is less than the node point on the axis
        if query_point[axis] < node.point[axis]:
            best = self.nearest_neighbor_search(query_point, node.left, depth+1, best)
            far_node = node.right
        else:
            best = self.nearest_neighbor_search(query_point, node.right, depth+1, best)
            far_node = node.left

        axis_distance = abs(query_point[node.axis] - node.point[node.axis])
        if axis_distance < best[1]:
            best = self.nearest_neighbor_search(query_point, far_node, depth+1, best)

        return best

    #eliminates recursion by using a stack
    def nearest_neighbor_search_iterative(self, query_point, root=None):
        return _nearest_neighbor_search_iterative(query_point, root, self.p1_points)

    

    @time_function
    def batch_distances(self, query_points, target_tree):
        n_query_points = query_points.shape[0]
        nearest_neighbors = np.zeros_like(query_points)
        distances = np.zeros(n_query_points)


        for i in range(n_query_points):
            if i % 2 == 0:
                print(f"Processing query point {i} of {n_query_points}")
            nearest_neighbors[i], distances[i] = self.nearest_neighbor_search_iterative(query_points[i], target_tree)
        return nearest_neighbors, distances

    
    @time_function
    def parallel_batch_distances(self, query_points, target_tree,tree_points_array, n_jobs=4):
        # Pass the numpy array instead of self to avoid pickling Open3D objects
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_nearest_neighbor_search_iterative)(p, target_tree, tree_points_array) for p in query_points
        )
        neighbors, dists = zip(*results)
    
        return np.array(neighbors), np.array(dists)
    



        

               

