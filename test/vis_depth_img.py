import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d


def visualize_row(depth_image_path, row_idx):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 提取指定行的像素值
    row = depth_image[row_idx, :]

    # 可视化指定行的像素值
    plt.plot(row)
    plt.xlabel('列')
    plt.ylabel('深度值')
    plt.title('指定行的深度值可视化')
    plt.show()

def visualize_column(depth_image_path, column_idx):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    
    # 提取指定行的像素值
    column = depth_image[:,column_idx]

    # 可视化指定行的像素值
    plt.plot(column)
    plt.xlabel('行')
    plt.ylabel('深度值')
    plt.title('指定行的深度值可视化')
    plt.show()

def visualize_depth_image(depth_image_path):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 可视化深度图像
    plt.imshow(depth_image, cmap='gray')
    plt.colorbar()
    plt.xlabel('列')
    plt.ylabel('行')
    plt.title('深度图像可视化')
    plt.show()


def visualize_depth_image_colormap(depth_image_path):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 将深度值映射到伪彩色图像
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # 可视化伪彩色深度图像
    plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.xlabel('列')
    plt.ylabel('行')
    plt.title('伪彩色深度图像可视化')
    plt.show()

def visualize_depth_image_pointcloud(depth_image_path):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 提取深度图像中的点坐标和深度值
    rows, cols = depth_image.shape
    y, x = np.meshgrid(range(rows), range(cols))
    x = x.flatten()
    y = y.flatten()
    z = depth_image[y, x]

    # 设置点云点的坐标和颜色
    point_cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    point_cloud.colors = o3d.utility.Vector3dVector(np.column_stack((z, z, z)))

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])

def visualize_depth_image_as_mountain(depth_image_path):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 创建网格
    rows, cols = depth_image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    # 将深度值作为山峰高度
    Z = depth_image.astype(float)
    
    # 可视化深度图像山峰
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='terrain')
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    ax.set_zlabel('深度值')
    ax.set_title('深度图像山峰可视化')
    plt.show()

def visualize_depth_image_contour(depth_image_path):
    # 读取深度图像文件
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 创建网格
    rows, cols = depth_image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    # 将深度值作为等高线高度
    Z = depth_image.astype(float)

    # 可视化深度图像等高线
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, cmap='terrain', levels=100)
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    ax.set_title('深度图像等高线可视化')
    plt.colorbar(contour)
    plt.show()


def main():
    depth_image_path = r'E:\realman-robot\test\test_images\lock_depth_image_1.png'
    # visualize_row(depth_image_path, row_idx=400)
    # visualize_column(depth_image_path, column_idx=950)
    visualize_depth_image(depth_image_path)
    # visualize_depth_image_colormap(depth_image_path)
    # visualize_depth_image_pointcloud(depth_image_path)
    # visualize_depth_image_as_mountain(depth_image_path)
    # visualize_depth_image_contour(depth_image_path)
    # visualize_depth_image_smooth(depth_image_path)

main()