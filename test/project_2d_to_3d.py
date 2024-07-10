'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-17 18:53:37
Version: v1
File: 
Brief: 
'''
import numpy as np
import cv2

def project_2d_to_3d(point_2d, depth_img_path, intrinsic_matrix):
        """
        Projects a 2D point on the RGB image to 3D space using the depth image and camera intrinsics.

        Args:
            point_2d: A 2D point as a list or tuple [x, y].
            depth_img_path: Path to the depth image (PNG).
            intrinsic_matrix: Camera intrinsics matrix (3x3 numpy array).

        Returns:
            A 3D point as a list [x, y, z].
        """

        # Load the depth image
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        # Get camera intrinsics
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        # Get depth value at the 2D point
        # print(depth_image[0,0])
        depth_value = depth_image[round(point_2d[1]), round(point_2d[0])] / 1000.0  # Convert depth from mm to meters

        # for i in range(1069,1075):
        #      print('\n')
        #      for j in range(465,478):
        #           print(depth_image[j,i],end=' ')

        # Check for invalid depth (0)
        if depth_value == 0:
            print('depth value is zero')
            return None  # Or handle invalid depth as needed

        # Calculate 3D coordinates
        x = (point_2d[0] - cx) * depth_value / fx
        y = (point_2d[1] - cy) * depth_value / fy
        z = depth_value
        print(x,y,z)
        return [x, y, z]

def test():
    # point_2d = [1096.8002844051193,457.9704534681624]#[864.361811155914,437.7130376344086]#[817.5269739192569,395.8120757413362]
    # point_2d = [1069,475] # for handle_depth_image_demo2
    point_2d = [937.1883239171375,642.846516007533] # for lock_depth_image_1
    # depth_img_path = '/media/datadisk10tb/leo/projects/realman-robot/test/test_images/handle_depth_image_demo1.png'
    depth_img_path = r'E:\realman-robot\test\test_images\lock_depth_image_1.png'
    # intrinsic_matrix = np.array([[607.9666748046875, 0, 320.0655822753906], [0, 607.6690063476562, 247.3187255859375], [0, 0, 1]])
    # intrinsic_matrix = np.array([[897.5597682,0,634.26241129],[0,896.33836351,381.57215086],[0,0,1]])
    # intrinsic_matrix = np.array([[897.6354,0,637.0919],[0,896.4505,371.6496],[0,0,1]]) # by matlab 2020
    # intrinsic_matrix = np.array([[902.9446,0,637.6256],[0,903.3026,366.0768],[0,0,1]]) # by matlab 2020
    intrinsic_matrix = np.array([[911.9500122070312,0,640.0983276367188],[0,911.5035400390625,370.97808837890625],[0,0,1]]) # by realsense api

    project_2d_to_3d(point_2d, depth_img_path, intrinsic_matrix)
    # 0.8133284993833853 0.2428993226783411 0.994
    # 0.46196093592231313 0.16167266059444962 0.516
    # 0.7451152781022438 0.29782234295874915 0.663
if __name__ == "__main__":
     test()