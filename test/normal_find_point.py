import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_point_B(point_A, normal, distance):
    normal = np.array(normal)
    normal /= np.linalg.norm(normal)  # Normalize the normal vector
    point_B = point_A + distance * normal
    return point_B

def project_3d_to_2d(point_A,intrinsic_matrix):
    projection_matrix = np.column_stack((intrinsic_matrix, [0, 0, 0]))
    point_A_homogeneous = np.array([point_A[0], point_A[1], point_A[2], 1])
    projected_point_A = np.dot(projection_matrix, point_A_homogeneous)
    projected_point_A /= projected_point_A[2]
    projected_point_A = projected_point_A[:2].astype(int)
    print(f'projected_point_A:{projected_point_A}')
    return projected_point_A


def visualize_points_and_normal(point_A, point_B, normal_center, normal_end, rgb_image_path, intrinsic_matrix,save_path):

    # Load RGB image and get image size 
    rgb_image = cv2.imread(rgb_image_path)
    height, width, _ = rgb_image.shape

    # Project 3D points to 2D image coordinates
    projected_point_A = project_3d_to_2d(point_A,intrinsic_matrix)
    projected_point_B = project_3d_to_2d(point_B,intrinsic_matrix)
    projected_normal_center = project_3d_to_2d(normal_center,intrinsic_matrix)
    projected_normal_end = project_3d_to_2d(normal_end,intrinsic_matrix)

    # Visualize points on the RGB image
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    
    plt.plot([projected_point_A[0], projected_point_B[0]], [projected_point_A[1], projected_point_B[1]], 'r')
    plt.scatter(projected_point_A[0], projected_point_A[1], c='r', marker='o')
    plt.scatter(projected_point_B[0], projected_point_B[1], c='r', marker='o')

    # Add point labels
    plt.text(projected_point_A[0] + 10, projected_point_A[1] + 10, 'A, handle center', color='b', fontsize=8)
    plt.text(projected_point_B[0] + 10, projected_point_B[1] + 10, 'B', color='b', fontsize=12)

    # Visualize normal on the RGB image
    plt.plot([projected_normal_center[0], projected_normal_end[0]], [projected_normal_center[1], projected_normal_end[1]], 'r')
    plt.scatter(projected_normal_center[0], projected_normal_center[1], c='r', marker='o')
    # plt.scatter(projected_normal_end[0], projected_normal_end[1], c='r', marker='o')
    
    # Add the normal vector as an arrow
    plt.arrow(projected_normal_center[0], projected_normal_center[1], projected_normal_end[0] - projected_normal_center[0], projected_normal_end[1] - projected_normal_center[1],
              color='b', length_includes_head=True, head_width=30, head_length=30)
    
    # Add point labels
    plt.text(projected_normal_center[0] + 10, projected_normal_center[1] + 10, 'door center', color='b', fontsize=12)
    # plt.text(projected_normal_end[0] + 10, projected_normal_end[1] + 10, 'B', color='b', fontsize=12)

    plt.title('Points A, B, and Normal Visualization')
    plt.savefig(save_path)
    plt.show()

def main():
    # calculate_point_B
    point_A = [0.37436895759697336,0.07038932309728949,0.731] # handle_center
    normal = [-0.02982215,0.47017003,-0.88207187]
    distance = 0.10
    point_B = calculate_point_B(point_A,normal,distance)
    print(f'point_B:{point_B}') # [0.36989564 0.14091483 0.59868922]

    normal_center = [0.01627231,0.0125639,0.70988184]
    normal = [-0.02982215,0.47017003,-0.88207187]
    distance = 0.20
    normal_end = calculate_point_B(normal_center,normal,distance)

    # visualize_points_and_normal
    rgb_image_path = '/media/datadisk10tb/leo/projects/realman-robot/test/test_images/handle_rgb_image_demo1.png'
    intrinsic_matrix = [[897.6354,0,637.0919],[0,896.4505,371.6496],[0,0,1]]
    save_path = '/media/datadisk10tb/leo/projects/realman-robot/test/test_images/handle_rgb_image_demo1_with_AB.png'
    
    visualize_points_and_normal(point_A, point_B, normal_center, normal_end, rgb_image_path, intrinsic_matrix, save_path)

if __name__ == "__main__":
    main()