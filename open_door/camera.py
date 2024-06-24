#!/usr/bin/env python

import numpy as np
import cv2
import time
import pyrealsense2 as rs
import keyboard
import json
import matplotlib.pyplot as plt

class CamIntrinsic(object):
    def __init__(self,intrinsic):
        self.fx = intrinsic[0]
        self.fy = intrinsic[4]
        self.cx = intrinsic[6]
        self.cy = intrinsic[7]
        self.intrinsic_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    def __str__(self):
        return f"CamIntrinsic(\n  fx={self.fx},\n  fy={self.fy},\n  cx={self.cx},\n  cy={self.cy},\n  intrinsic_matrix=\n{self.intrinsic_matrix}\n)"

class Camera(object):

    def __init__(self,cam_params_path=None,fps=30):
        with open(cam_params_path, 'r') as f:
            data = json.load(f)
        self.width = data['width']
        self.height = data['height']
        self.intrinsic = CamIntrinsic(data['intrinsic_matrix'])
        self.depth_scale = data['depth_scale']
        # connect
        self.connect(fps)

    def __str__(self):
        return f"RealSense(\n  width={self.width},\n  height={self.height},\n  {self.intrinsic.__str__()},\n  depth_scale={self.depth_scale}\n)"

    def connect(self,fps=30):
        print('==========\nCamera Connecting...')
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_device('238122071696')
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        print('Camera Connected\n==========')

    def disconnect(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def init_intrinsic(self):
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print(intrinsics)
        print("intrinsics:")
        print(f"Width: {intrinsics.width}")
        print(f"Height: {intrinsics.height}")
        print(f"FX: {intrinsics.fx}")
        print(f"FY: {intrinsics.fy}")
        print(f"CX: {intrinsics.ppx}")
        print(f"CY: {intrinsics.ppy}")
        print("distortion coefficient:")
        print(f"K1: {intrinsics.coeffs[0]}")
        print(f"K2: {intrinsics.coeffs[1]}")
        print(f"P1: {intrinsics.coeffs[2]}")
        print(f"P2: {intrinsics.coeffs[3]}")
        print(f"K3: {intrinsics.coeffs[4]}")
        return CamIntrinsic([intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy])

    def init_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return depth_scale

    def capture_rgb(self,rgb_save_path=None):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        rgb_img = np.asanyarray(color_frame.get_data())
        if rgb_save_path is not None:
            cv2.imwrite(rgb_save_path,rgb_img)
        return rgb_img

    def capture_d(self,d_save_path=None):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        d_img = np.asanyarray(aligned_depth_frame.get_data())
        if d_save_path is not None:
            cv2.imwrite(d_save_path,d_img)
        return d_img

    def capture_rgbd(self,rgb_save_path=None,d_save_path=None):
        frames = self.pipeline.wait_for_frames()
        # color = frames.get_color_frame()
        # depth = frames.get_depth_frame()
        # rgb_img=np.asarray(color.get_data())
        # d_img=np.asarray(depth.get_data())
        # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        d_img = np.asanyarray(aligned_depth_frame.get_data())
        rgb_img = np.asanyarray(color_frame.get_data())
        if rgb_save_path is not None:
            cv2.imwrite(rgb_save_path,rgb_img)
        if d_save_path is not None:
            cv2.imwrite(d_save_path,d_img)
        return rgb_img,d_img

    def capture_video(self,duration,fps,save_path=None):
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(save_path, fourcc, fps, (self.width, self.height))
        # Record for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert depth and color frames to OpenCV images
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # # Write combined depth and color image (optional)
            # combined_image = np.hstack((depth_image, color_image))
            # out.write(combined_image)
            # Write only color image
            out.write(color_image)
        out.release()

    def vis_rgbd(self,d_img_path, rgb_img_path=None, save_path="depth_visualization.png"):
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
            # print(f"Depth visualization saved to: {save_path}")


    def normal2rxryrz(self,normal,if_p=False):
        from scipy.spatial.transform import Rotation as R
        original_normal = np.array(normal)
        normal = original_normal * -1
        z_axis = normal / np.linalg.norm(normal)
        initial_x_axis = np.array([1, 0, 0])
        x_axis = initial_x_axis - np.dot(initial_x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
        rx,ry,rz = euler_angles
        if if_p:
            print(f'original_normal:\n{original_normal}')
            print(f'normal:\n{normal}')
            print(f'z_axis:\n{z_axis}')
            print(f'initial_x_axis:\n{initial_x_axis}')
            print(f'x_axis:\n{x_axis}')
            print(f'y_axis:\n{y_axis}')
            print(f'rotation_matrix:\n{rotation_matrix}')
            print(f'euler_angles:\n{euler_angles}')
            print(f'rx:{rx} ry:{ry} rz:{rz}')
        return rx,ry,rz

    def vis_d(self,d_img_path,save_path,show=False):
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
            # print(f"Depth visualization saved to: {save_path}") 
        if show:
            plt.show()

    def xy_depth_2_xyz(self,u,v,depth):
        fx = self.intrinsic.fx
        fy = self.intrinsic.fy
        cx = self.intrinsic.cx
        cy = self.intrinsic.cy
        x = (u - cx) * depth * self.depth_scale / fx
        y = (v - cy) * depth * self.depth_scale / fy
        z = depth * self.depth_scale
        return x, y, z

    def xy2xyz(self, u, v, d_img, radius=5, depth_threshold=0.02, valid_ratio_threshold=0.60):
        """
        Converts a pixel point (u, v) to 3D coordinates (x, y, z) using a depth image.
        Handles potential zero-depth values and applies averaging for robustness.

        Args:
            u (float): x-coordinate of the pixel.
            v (float): y-coordinate of the pixel.
            d_img (str or np.ndarray): Path to depth image or the depth image itself.
            radius (int): Radius around the pixel to consider for averaging.
            depth_threshold (float): Maximum depth difference between neighboring pixels to be 
                                     considered valid (in meters).
            valid_ratio_threshold (float): Minimum ratio of valid depth values within the 
                                           averaging region. 

        Returns:
            tuple: (x, y, z) coordinates in meters, or None if depth estimation is unreliable.
        """

        fx = self.intrinsic.fx
        fy = self.intrinsic.fy
        cx = self.intrinsic.cx
        cy = self.intrinsic.cy

        if isinstance(d_img, str):
            d_img = cv2.imread(d_img, cv2.IMREAD_UNCHANGED)

        # 1. Extract Region of Interest (ROI)
        u, v = int(u), int(v)  # Ensure integer indices
        height, width = d_img.shape[:2]
        u_min, u_max = max(0, u - radius), min(width - 1, u + radius)
        v_min, v_max = max(0, v - radius), min(height - 1, v + radius)
        depth_roi = d_img[v_min:v_max+1, u_min:u_max+1]

        # 2. Filter for Valid Depths
        center_depth = np.mean(depth_roi[depth_roi != 0]) # center_depth = depth_roi[radius, radius]
        valid_depth_mask = (np.abs(depth_roi - center_depth) * self.depth_scale <= depth_threshold) & (depth_roi != 0)
        
        # 3. Check for Sufficient Valid Data
        valid_ratio = np.sum(valid_depth_mask) / np.count_nonzero(depth_roi)
        if valid_ratio < valid_ratio_threshold:
            print(f"ERROR: Not enough valid depth values around the point. Ratio: {valid_ratio:.2f}")
            return None 

        # 4. Calculate Average Depth 
        average_depth = np.mean(depth_roi[valid_depth_mask])

        # 5. Convert to 3D Coordinates
        x = (u - cx) * average_depth * self.depth_scale / fx
        y = (v - cy) * average_depth * self.depth_scale / fy
        z = average_depth * self.depth_scale

        return x, y, z, average_depth

    def create_point_cloud_from_depth_image(self, depth, organized=True):
        """ Generate point cloud using depth image only.

            Input:
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    depth image
                organized: bool
                    whether to keep the cloud in image shape (H,W,3)

            Output:
                cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                    generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        assert (depth.shape[0] == self.height and depth.shape[1] == self.width)
        xmap = np.arange(self.width)
        ymap = np.arange(self.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / self.depth_scale
        points_x = (xmap - self.intrinsic.cx) * points_z / self.intrinsic.fx
        points_y = (ymap - self.intrinsic.cy) * points_z / self.intrinsic.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        return cloud

    def check_rs_resolution(self):
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()

        for profile in depth_sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.depth:
                width, height = profile.as_video_stream_profile().width(), profile.as_video_stream_profile().height()
                print(f"Depth Stream Resolution: {width} x {height}")

        for profile in device.query_sensors()[1].get_stream_profiles():
            if profile.stream_type() == rs.stream.color:
                width, height = profile.as_video_stream_profile().width(), profile.as_video_stream_profile().height()
                print(f"Color Stream Resolution: {width} x {height}")

    def get_serial_num(self):
        devices = rs.context().query_devices()
        for dev in devices:
            serial_number = dev.get_info(rs.camera_info.serial_number)
            print(f"Device: {serial_number}")

    def display_and_record(self):
        cv2.namedWindow('RealSense RGB', cv2.WINDOW_AUTOSIZE)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('realsense_clip.avi', fourcc, 30.0, (1280, 720))
        recording = False
        frame_count = 0
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow('RealSense RGB', color_image)

                # Record if 'r' is pressed
                if keyboard.is_pressed('r'):
                    recording = True
                    print("Recording started.")

                # Pause recording if 'p' is pressed
                if keyboard.is_pressed('p'):
                    recording = False
                    print("Recording paused.")

                # Save frame to video if recording is enabled
                if recording:
                    out.write(color_image)
                    frame_count += 1

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            out.release()
            print(f"Recording stopped. {frame_count} frames recorded.")

if __name__ == "__main__":
    image_dir = './images/image1/'
    camera = Camera(f'cam_params_matlab_2.json')

    print(camera)
    
    # capture images
    time.sleep(3)
    rgb_img,d_img = camera.capture_rgbd(rgb_save_path=f'{image_dir}/rgb.png',d_save_path=f'{image_dir}/d.png')

    # disconnect
    camera.disconnect()