import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pclouder.pclouder import PClouder





def main():
    pcd1 = o3d.io.read_point_cloud("pcloud/scene1_perspective1.ply")
    pcd2 = o3d.io.read_point_cloud("pcloud/scene1_perspective2.ply")
    pclouder = PClouder(pcd1, pcd2)
    pclouder.show_side_by_side()




if __name__ == "__main__":
    main()