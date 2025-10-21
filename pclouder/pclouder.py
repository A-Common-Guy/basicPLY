import open3d as o3d
import numpy as np
class PClouder:

    '''
    Align the two perspectives with custom methods
    everything with numpy
    '''
    def __init__(self, perspective_1_ply, perspective_2_ply):
        self.p1 = perspective_1_ply
        self.p2 = perspective_2_ply
        self.p1_points = np.asarray(self.p1.points)
        self.p2_points = np.asarray(self.p2.points)

        self.initial_guess_transform = np.eye(4)


    def set_initial_guess_transform(self, initial_guess_transform):
        self.initial_guess_transform = initial_guess_transform


    def show_side_by_side(self):
        o3d.visualization.draw_geometries([self.p1, self.p2])

    def save_pcloud(self, pcloud, pcloud_path):
        o3d.io.write_point_cloud(pcloud_path, pcloud)


    def apply_icp(self,initial_guess_transform):
