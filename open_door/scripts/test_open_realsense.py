#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import pyrealsense2 as rs

cam0_path = r'/home/rm/catkin_ws/images/' # 提前建立好的存储照片文件的目录


def test_realsenseD435():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 1280*720
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    scaling_factor = 2.0
    cv_img = cv2.resize(color_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    # cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video
    cv2.imwrite(cam0_path + '/' + 'test1' + '.jpg', cv_img)
    pipeline.stop()
    cv2.destroyAllWindows()

def get_intrinsics():
    # 创建深度和颜色流的配置
    config = rs.config()
    config.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 0, 0, rs.format.bgr8, 30)

    # 启动 RealSense 管道
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # 获取深度和颜色流的相机参数
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # 打印相机参数
    print("内参矩阵:")
    print(f"Width: {intrinsics.width}")
    print(f"Height: {intrinsics.height}")
    print(f"FX: {intrinsics.fx}")
    print(f"FY: {intrinsics.fy}")
    print(f"CX: {intrinsics.ppx}")
    print(f"CY: {intrinsics.ppy}")
    print("畸变参数:")
    print(f"K1: {intrinsics.coeffs[0]}")
    print(f"K2: {intrinsics.coeffs[1]}")
    print(f"P1: {intrinsics.coeffs[2]}")
    print(f"P2: {intrinsics.coeffs[3]}")
    print(f"K3: {intrinsics.coeffs[4]}")
    # 停止管道并关闭窗口
    pipeline.stop()


def capture_images_jpg():
    cam0_path = r'./test_images/'  # Directory to store the captured photos
    num_frames_to_skip = 10  # Number of initial frames to skip for stability
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)

    # Skip initial frames for stability
    for _ in range(num_frames_to_skip):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Save RGB image
    cv2.imwrite(cam0_path + 'monitor_rgb_image.jpg', color_image)

    # Adjust depth image for visualization (optional)
    depth_image_visualized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Save depth image
    cv2.imwrite(cam0_path + 'monitor_depth_image.jpg', depth_image_visualized)

    pipeline.stop()

def capture_images_png():
    cam0_path = r'../src/image/'  # Directory to store the captured photos
    num_frames_to_skip = 100  # Number of initial frames to skip for stability

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)

    # Skip initial frames for stability
    for _ in range(num_frames_to_skip):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Save RGB image as PNG
    cv2.imwrite(cam0_path + 'rgb_handle_1.png', color_image)

    # Adjust depth image for visualization (optional)
    # depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Save depth image as PNG
    cv2.imwrite(cam0_path + 'd_handle_1.png', depth_image)

    pipeline.stop()

def test_check_rs():
    # 枚举可用设备
    ctx = rs.context()
    devices = ctx.query_devices()

    # 打印已连接设备的信息
    if len(devices) > 0:
        print("已连接的RealSense设备：")
        for dev in devices:
            print(f"设备名称: {dev.get_info(rs.camera_info.name)}")
            print(f"序列号: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"传感器类型: {dev.get_info(rs.camera_info.product_id)}")
    else:
        print("未检测到已连接的RealSense设备")

def display():
    # Configure color stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create OpenCV window
    cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Wait for a color frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Show image
            cv2.imshow('RealSense RGB', color_image)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


def record_video():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    # Create a VideoWriter object to save the video
    # Adjust the filename, codec, and other parameters as needed
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

    try:
        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()

            # Get the color frame
            color_frame = frames.get_color_frame()

            # Convert the frame to a numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Write the frame to the video file
            out.write(color_image)

    except KeyboardInterrupt:  # Stop recording when Ctrl+C is pressed
        pass

    finally:
        # Stop the pipeline and release resources
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    capture_images_png()
    # display()
    # record_video()
