'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-14 17:43:42
Version: v1
File: 
Brief: 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def vis_rgbd(d_img_path, rgb_img_path=None, save_path="depth_visualization.png"):
    # Load the depth image
    depth_img = cv2.imread(d_img_path, cv2.IMREAD_UNCHANGED)

    # Create a mask where zero-depth pixels are True (white)
    zero_depth_mask = depth_img == 0

    # Create an empty image (initially black)
    visualized_depth = np.zeros_like(depth_img, dtype=np.uint8)

    # Set zero-depth pixels to white
    visualized_depth[zero_depth_mask] = 255

    # Optional: Blend with RGB image for better visualization
    if rgb_img_path is not None:
        rgb_img = cv2.imread(rgb_img_path)
        # Resize RGB image to match depth image dimensions if needed
        if rgb_img.shape[:2] != depth_img.shape[:2]:
            rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]))
        
        # ----> Convert depth image to 3 channels:
        visualized_depth = cv2.cvtColor(visualized_depth, cv2.COLOR_GRAY2BGR)

        # Blend images (adjust alpha for desired transparency)
        alpha = 0.5  # Example: 50% transparency
        visualized_depth = cv2.addWeighted(visualized_depth, alpha, rgb_img, 1 - alpha, 0)

    # Save the visualized depth image
    if save_path:
        cv2.imwrite(save_path, visualized_depth)
        print(f"Depth visualization saved to: {save_path}")


def vis_d(d_img_path,save_path,show=False):
    # Load the depth image
    depth_image = cv2.imread(d_img_path, cv2.IMREAD_ANYDEPTH)

    # Check if the image is loaded correctly
    if depth_image is None:
        print("Error: Could not load the depth image.")
        exit()

    # Normalize the depth image to 0-255 range for visualization
    output_displayable = cv2.normalize(depth_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_displayable = cv2.cvtColor(output_displayable, cv2.COLOR_GRAY2BGR)

    # Display the depth image using matplotlib
    plt.imshow(output_displayable)
    plt.title("Depth Image")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)  # Save the image
        print(f"Depth visualization saved to: {save_path}") 
    if show:
        plt.show()

if __name__ == "__main__":
    d_img_path = r"E:\realman-robot-2\open_door\images\image14\d.png"
    rgb_img_path = r"E:\realman-robot-2\open_door\images\image14\rgb.png" # Optional
    save_path = 'rgbd_visualization.png'
    vis_rgbd(d_img_path, rgb_img_path, save_path)
    d_img_path = r"E:\realman-robot-2\open_door\images\image14\d.png"
    save_path = 'd_visualization.png'
    vis_d(d_img_path,save_path,show=True)