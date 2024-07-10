'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-25 17:07:05
Version: v1
File: 
Brief: 
'''
import cv2
import numpy as np
import open3d as o3d

def get_handle_center(rgb_image_path, depth_image_path, intrinsics_path):
    # Load RGB image
    rgb_image = cv2.imread(rgb_image_path)

    # Load depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

    # Load camera intrinsics
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)

    # Convert depth image to point cloud
    depth_scale = 1000.0  # Depth scale factor (depends on how the depth image was captured)
    depth_image = depth_image.astype(np.float32) / depth_scale
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        o3d.geometry.Image(depth_image),
        depth_scale=depth_scale,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    # Generate handle mask
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_threshold = np.array([0, 0, 0])  # Adjust the lower threshold for handle color
    upper_threshold = np.array([255, 255, 255])  # Adjust the upper threshold for handle color
    handle_mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

    # Find handle center
    handle_center = np.array([0.0, 0.0, 0.0])
    num_points = 0

    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            if handle_mask[i, j] > 0:
                point = pcd.get_point(j, i)
                handle_center += point
                num_points += 1

    if num_points > 0:
        handle_center /= num_points

    return handle_center

# Example usage
rgb_image_path = 'rgb_image.jpg'
depth_image_path = 'depth_image.png'
intrinsics_path = 'intrinsics.json'

handle_center = get_handle_center(rgb_image_path, depth_image_path, intrinsics_path)
print("Handle Center (Camera Coordinates):", handle_center)