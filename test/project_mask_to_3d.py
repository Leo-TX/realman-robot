'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-07 16:02:09
Version: v1
File: 
Brief: 
'''

import imageio
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def calculate_mean_depth(mask_path, depth_path):
    # Read the mask image
    mask_image = imageio.imread(mask_path)

    # Read the depth image
    depth_image = imageio.imread(depth_path)

    # Convert mask image to a binary mask (0s and 1s)
    mask = np.where(mask_image > 0, 1, 0)

    # Extract depth values corresponding to masked areas
    masked_depth_values = depth_image[mask == 1]

    # Calculate the mean depth value
    mean_depth = np.mean(masked_depth_values)

    return mean_depth

def visualize_masked_depth(mask_path, depth_path):
    # Read the mask image
    mask_image = imageio.imread(mask_path)

    # Read the depth image
    depth_image = imageio.imread(depth_path)

    # Convert mask image to a binary mask (0s and 1s)
    mask = np.where(mask_image > 0, 1, 0)

    # Create a masked depth image
    masked_depth_image = np.where(mask == 1, depth_image, 0)

    # Display the masked depth image
    plt.imshow(masked_depth_image, cmap='jet')
    plt.colorbar()
    plt.show()

def project_2d_to_3d(point_2d, depth_value, intrinsic_matrix):
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        x = (point_2d[0] - cx) * depth_value / fx
        y = (point_2d[1] - cy) * depth_value / fy
        z = depth_value
        print(x,y,z)
        return [x, y, z]

def main():
    mask_file = r'E:\realman-robot\test\test_images\lock_rgb_image_1_mask.png'
    depth_file = r'E:\realman-robot\test\test_images\lock_depth_image_test.png'
    depth_value = calculate_mean_depth(mask_file, depth_file)
    print("Mean Depth Value:", depth_value)
    visualize_masked_depth(mask_file, depth_file)

    point_2d = [937.1883239171375,642.846516007533] # for lock_depth_image_1
    intrinsic_matrix = np.array([[911.9500122070312,0,640.0983276367188],[0,911.5035400390625,370.97808837890625],[0,0,1]]) # by realsense api
    project_2d_to_3d(point_2d, depth_value, intrinsic_matrix)

if __name__ == "__main__":
    main()