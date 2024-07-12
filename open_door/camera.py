import numpy as np
import cv2
import time
import pyrealsense2 as rs
import keyboard
import json
import matplotlib.pyplot as plt

from utils.lib_io import *

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
    def __init__(self,width=1280,height=720,intrinsic_matrix=None,depth_scale=0.001,fps=30):
        self.width = width
        self.height = height
        self.intrinsic = CamIntrinsic(intrinsic_matrix)
        self.depth_scale = depth_scale

        self.connect(fps)

    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_cam.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.width,cfg.height,cfg.intrinsic_matrix,cfg.depth_scale,cfg.fps)

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
                    out = cv2.VideoWriter('realsense_clip.avi', fourcc, 30.0, (1280, 720))
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
    camera = Camera.init_from_yaml(cfg_path='cfg/cfg_cam.yaml')
    print(camera)
    camera.display_and_record()