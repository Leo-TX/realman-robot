'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-30 16:43:14
Version: v1
File: 
Brief: 
'''
#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import os
import pyrealsense2 as rs
import sys
sys.path.append('../..')
from arm import Arm

cam0_path = r'E:\realman-robot-2\open_door\arm_package\handeye_calibration\data\2024061301\\'
if not os.path.exists(cam0_path):
    os.makedirs(cam0_path)

def callback(frame):
    scaling_factor = 1.0
    global count
    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Capture_Video", cv_img)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('s'):
        pose = arm.get_p()
        print(f'pose_{count}: {pose}')
        with open(f'{cam0_path}\poses.txt', 'a+') as f:
            pose = [str(i) for i in pose]
            new_line = f'{",".join(pose)}\n'
            f.write(new_line)
        cv2.imwrite(f"{cam0_path}\\{str(count)}.jpg" , cv_img)
        count += 1
    else:
        pass

def collect_data():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    global count
    count = 1
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    arm = Arm('192.168.10.19',8080)
    arm.change_tool_frame('dh3',if_p=True)
    collect_data()
