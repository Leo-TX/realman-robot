'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-17 19:21:22
Version: v1
File: 
Brief: 
'''
import numpy as np
import time
import cv2
import sys
import os
import shutil
import json
import torch
import threading
from collections import namedtuple
from matplotlib import pyplot as plt

from arm import Arm
from base import Base
from camera import Camera
from head import Head
from dtsam import DTSAM
from server import Server
from ransac import RANSAC
from dmp import DMP
from _primitive import _Primitive

from utils.math import *
from utils.lib_io import *

## for safty
GRASP_CURRENT_THRESHOLD_L = -15000
GRASP_CURRENT_THRESHOLD_H = 15000

UNLOCK_CURRENT_THRESHOLD_L = -18000
UNLOCK_CURRENT_THRESHOLD_H = 18000

ROTATE_CURRENT_THRESHOLD_L = -18000
ROTATE_CURRENT_THRESHOLD_H = 18000

OPEN_CURRENT_THRESHOLD_L = -25000
OPEN_CURRENT_THRESHOLD_H = 25000

PULL_DOOR_THRESHOLD_JOINT_4_L = -10000
PUSH_DOOR_THRESHOLD_JOINT_4_H = 15000

## Primitive Types
# DUAL = 0 # [0/1,0,0] # 0 for right 1 for left

HOME = 0 # [0,0,0]
PREMOVE = 1 # [T,0,0]
GRASP = 2 # [dx,dy,dz]
UNLOCK = 3 # [T,0,0]
ROTATE = 4 # [T,0,0]
OPEN = 5 # [T,0,0]
START = 6
FINISH = 7
BACK = 8
CLEAR = 9

## Error Types
SUCCESS = 1

SAFTY_ISSUE = 2131
NO_SAFTY_ISSUE = 3121

# PREMOVE_TOO_CLOSE = -1

GRASP_NO_HANDLE = -1
GRASP_IK_FAIL = -1
GRASP_SAFTY = 0
GRASP_MISS = -1

ROTATE_MISS = -1
ROTATE_SAFTY = 0
ROTATE_IK_FAIL = -1

UNLOCK_MISS = -1
UNLOCK_SAFTY = 0
UNLOCK_IK_FAIL = -1

OPEN_MISS = -1
OPEN_SAFTY = 0
OPEN_FAIL = -1

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time() 
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time 
        print(f"[Time] {func.__name__} execution time: {execution_time:.4f} s")
        return result
    return wrapper

class Primitive(object):
    def __init__(self,cfg_path='./cfg/cfg.yaml',root_dir='./',tjt_num=1):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        self.cfg = cfg

        ## init two arms
        arm_r = Arm.init_from_yaml(cfg_path=cfg.cfg_arm_right)
        arm_r = Arm.init_from_yaml(cfg_path=cfg.cfg_arm_left)
        ## init camera
        camera = Camera.init_from_yaml(cfg_path=cfg.cfg_cam)
        ## init base
        base = Base.init_from_yaml(cfg_path=cfg.cfg_base)
        ## init head
        head = Head.init_from_yaml(cfg_path=cfg.cfg_head)
        ## init server
        server = Server.init_from_yaml(cfg_path=cfg.cfg_server)
        ## init ransac
        ransac = RANSAC(cfg_ransac=cfg.cfg_ransac,cfg_cam=cfg.cfg_cam)
        ## init dtsam
        dtsam = DTSAM.init_from_yaml(cfg_path=cfg.cfg_dtsam)

        ## remote
        self.remote_python_path = cfg.remote_python_path
        self.remote_root_dir = cfg.remote_root_dir
        self.remote_img_dir = cfg.remote_img_dir

        ## os
        self.tjt_num = tjt_num
        self.action_num = 0
        self.root_dir = root_dir
        self.tjt_dir = f'{self.root_dir}/data/trajectory_{self.tjt_num:03d}/'
        if os.path.exists(self.tjt_dir):
             shutil.rmtree(self.tjt_dir)
        os.makedirs(self.tjt_dir)

        self.last_pmt = _Primitive(action="START",id=START,ret=1,param=[0,0,0],error="START")
        self.this_pmt = _Primitive(action="START",id=START,ret=1,param=[0,0,0],error="START")
        self.primitives = {0:self.last_pmt.to_list()}
        
        self.grasp_thresholds = [[GRASP_CURRENT_THRESHOLD_L, GRASP_CURRENT_THRESHOLD_H] for _ in range(6)]
        self.unlock_thresholds = [[UNLOCK_CURRENT_THRESHOLD_L, UNLOCK_CURRENT_THRESHOLD_H] for _ in range(6)]
        self.rotate_thresholds = [[ROTATE_CURRENT_THRESHOLD_L, ROTATE_CURRENT_THRESHOLD_H] for _ in range(6)]
        self.open_thresholds = [[OPEN_CURRENT_THRESHOLD_L, OPEN_CURRENT_THRESHOLD_H] for _ in range(6)]
        
    def disconnect_robot(self):
        print('========== Disconnecting... ==========')
        self.camera.disconnect()
        self.arm_r.disconnect()
        self.arm_l.disconnect()
        self.base.disconnect()
        self.head.disconnect()
        self.server.disconnect()
        print('========== Disconnected ==========')

    def __str__(self):
        return ''
    
    def action2num(self,action):
        if action == "dual":
            return DUAL
        elif action == "premove":
            return PREMOVE
        elif action == "grasp":
            return GRASP
        elif action == "unlock":
            return UNLOCK
        elif action == "rotate":
            return ROTATE
        elif action == "open":
            return OPEN
        elif action == "home":
            return HOME
        elif action == "finish":
            return FINISH
        elif action == "back":
            return BACK
        elif action == "clear":
            return CLEAR
        else:
            return -1

    def update(self,save_path=None):
        if save_path is None:
            save_path=f'{self.tjt_dir}/{self.action_num}.json'
        
        data = {"last_action": self.last_pmt.action,
                "last_id": self.last_pmt.id,
                "last_ret": self.last_pmt.ret,
                "last_param": self.last_pmt.param,
                "last_error": self.last_pmt.error,
                "this_action": self.this_pmt.action,
                "this_id": self.this_pmt.id,
                "this_ret": self.this_pmt.ret,
                "this_param": self.this_pmt.param,
                "this_error": self.this_pmt.error
                }
        
        with open(save_path,'w') as json_file:
            json.dump(data,json_file,indent=4)
        
        # self.this_pmt --> self.last_pmt
        for attr in ["action", "id", "ret", "param", "error"]:
            setattr(self.last_pmt, attr, getattr(self.this_pmt, attr))

        self.primitives[self.action_num] = self.this_pmt.to_list()
    
    def save_primitives(self,save_path=None):
        if save_path is None:
            save_path=f'{self.tjt_dir}/primitives.json'
        
        with open(save_path,'w') as json_file:
            json.dump(self.primitives,json_file,indent=4)

    # @time_it
    def capture(self,if_d=False,vis=False,if_update=True):
        # print('========== Image Capturing ... ==========')
        if if_update:
            self.action_num += 1
            self.rgb_img_path = f'{self.tjt_dir}/{self.action_num}.png'
        if if_d:
            self.d_img_path = f'{self.tjt_dir}/{self.action_num}/d.png'
            if not os.path.exists(os.path.dirname(self.d_img_path)):
                os.makedirs(os.path.dirname(self.d_img_path))
            rgb_img,d_img = self.camera.capture_rgbd(rgb_save_path=self.rgb_img_path,d_save_path=self.d_img_path)
            if vis:
                self.camera.vis_rgbd(d_img_path=self.d_img_path,rgb_img_path=self.rgb_img_path,save_path=f'{self.tjt_dir}/{self.action_num}/rgbd_vis.png')
                self.camera.vis_d(d_img_path=self.d_img_path,save_path=f'{self.tjt_dir}/{self.action_num}/d_vis.png')
            return rgb_img,d_img 
        else:
            if if_update:
                rgb_img = self.camera.capture_rgb(rgb_save_path=self.rgb_img_path)
            else:
                rgb_img = self.camera.capture_rgb(f'{self.tjt_dir}/temp.png')
            return rgb_img
        # print(f'========== Image Captured ==========')

    def start_current_monitor_thread(self, thresholds):
        self.current_data = {i: [] for i in range(7)}
        self.current_max = [0]*7
        self.current_min = [0]*7
        self.current_data_start_time = time.time()
        self.monitor_running = True
        self.current_monitor_thread = threading.Thread(target=self.current_monitor_loop, args=(thresholds,))
        self.current_monitor_thread.daemon = True
        self.current_monitor_thread.start()

    def current_monitor_loop(self, thresholds):
        while self.monitor_running:
            current_check_result = self.check_current_safety(thresholds)
            if current_check_result == 0:
                print(f"!!!SAFTY ISSUE!!!")
                self.this_pmt.ret = SAFTY_ISSUE
                self.this_pmt.error = 'SAFTY_ISSUE'
                self.arm.move_stop(if_p=True)
                break
            else:
                self.this_pmt.ret = NO_SAFTY_ISSUE
                self.this_pmt.error = 'NO_SAFTY_ISSUE'
            time.sleep(0.1)

    def check_current_safety(self, thresholds):
        current = self.arm.get_c()
        for i in range(7):
            self.current_data[i].append(current[i])
            self.current_max[i] = max(self.current_max[i],current[i])
            self.current_min[i] = min(self.current_min[i],current[i])
        for i, (min_current, max_current) in enumerate(thresholds):
            if current[i] < min_current or current[i] > max_current:
                return 0
        return 1

    def vis_current_data(self, save_path=None, show=False):
        if save_path is None:
            save_path = f'{self.tjt_dir}/{self.action_num}_current.png'
        plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        time_elapsed = [t - self.current_data_start_time for t in range(len(self.current_data[0]))] 
        for i in range(7):
            plt.plot(time_elapsed, self.current_data[i], label=f'Joint {i+1}', color=colors[i])
        plt.xlabel('Time')
        plt.ylabel('Current Value')
        plt.title('Current Data of Each Joint')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        if show:
            plt.show()

    @time_it
    def premove(self,premove_T):
        print(f'========== Premoving ... ==========')
        self.this_pmt.action = "PREMOVE"
        self.this_pmt.id = PREMOVE
        self.this_pmt.param = [premove_T,0,0]

        # self.base.move_T(T=premove_T)
        # time.sleep(2)

        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        if not os.path.exists(os.path.dirname(rgb_img_path)):
            os.makedirs(os.path.dirname(rgb_img_path))
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path

        self.normal,self.weights,self._3d_center,self._2d_center,self.mask_color = self.ransac.get_normal_server(rgb_img_path,d_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
        print(f'[RANSAC Result] normal: {self.normal} weights: {self.weights}')

        self.base.move_to_door(self.weights,offset_in_front=0.7,d2t_coefficient=4.8) # 4.8 for 1311LabDoor # 4.6 for 0311LabInsideDoor2
        time.sleep(2)

        self.this_pmt.ret = SUCCESS
        self.this_pmt.error = "NONE"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Premove Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def grasp(self,grasp_offset=[-0.04,0.03,0.01],thresholds=None):
        print('========== Grasping... ==========')
        self.this_pmt.action = "GRASP"
        self.this_pmt.id = GRASP
        self.this_pmt.param = grasp_offset

        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        if not os.path.exists(os.path.dirname(rgb_img_path)):
            os.makedirs(os.path.dirname(rgb_img_path))
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path
        
        ## dtsam
        print('DTSAM ...')
        self.x1_2d,self.y1_2d,self.orientation,self.w,self.h,self.box = self.dtsam.get_xy_server(rgb_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
        if self.w == 0 and self.h == 0:
            self.this_pmt.ret = GRASP_NO_HANDLE
            self.this_pmt.error = "GRASP_NO_HANDLE"
            print(f'[DTSAM Result] NO handle detections!!!')
        else:
            self.y1_2d -= 5 # for avoiding depth value error(zero)
            self.x2_2d,self.y2_2d = rotate_point(self.x1_2d,self.y1_2d,self.box,direction='counter-clockwise',angle=90)
            print(f'[center 2d point] x1_2d: {self.x1_2d}, y1_2d: {self.y1_2d}')
            print(f'[rotate 2d point] x2_2d: {self.x2_2d}, y2_2d: {self.y2_2d}')
            
            ## determin which arm
            if self.x1_2d < self.camera.width/2:
                self.arm = self.arm_l
                self.r_l = 'left'
                print(f'[Arm Choice]: LEFT')
            else:
                self.arm = self.arm_r
                self.r_l = 'right'
                print(f'[Arm Choice]: RIGHT')

            ## xy2xyz
            self.x1_3d,self.y1_3d,self.z1_3d,self.average_depth = self.camera.xy2xyz(self.x1_2d,self.y1_2d,d_img=d_img_path)
            self.x2_3d,self.y2_3d,self.z2_3d = self.camera.xy_depth_2_xyz(self.x2_2d,self.y2_2d,self.average_depth)
            print(f'[xy2xyz Result] x1_3d: {self.x1_3d}, y1_3d: {self.y1_3d}, z1_3d: {self.z1_3d}, average_depth: {self.average_depth}')
            print(f'[xy2xyz Result] x2_3d: {self.x2_3d}, y2_3d: {self.y2_3d}, z2_3d: {self.z2_3d}, average_depth: {self.average_depth}')

            ## ransac
            print('RANSAC ...')
            self.normal,self.weights,self._3d_center,self._2d_center,self.mask_color = self.ransac.get_normal_server(rgb_img_path,d_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
            print(f'[RANSAC Result] normal: {self.normal} weights: {self.weights}')

            ## normal2rxryrz
            self.rx,self.ry,self.rz = normal2rxryrz(self.normal)
            print(f'[normal2rxryrz Result] rx: {self.rx} ry: {self.ry} rz: {self.rz}')

            ## p1_3d_cam_xyzrxryrz 2 p1_3d_base_xyzrxryrz
            self.p1_3d_cam_xyzrxryrz = [self.x1_3d,self.y1_3d,self.z1_3d,self.rx,self.ry,self.rz]
            self.p2_3d_cam_xyzrxryrz = [self.x2_3d,self.y2_3d,self.z2_3d,self.rx,self.ry,self.rz]
            self.p1_3d_base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(self.p1_3d_cam_xyzrxryrz)
            self.p2_3d_base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(self.p2_3d_cam_xyzrxryrz)
            if self.orientation == 'horizontal':
                self.p1_3d_base_xyzrxryrz[4] -= np.pi/2
                self.p2_3d_base_xyzrxryrz[4] += np.pi
            print(f'[p1_3d_base_xyzrxryrz] {self.p1_3d_base_xyzrxryrz}')
            print(f'[p2_3d_base_xyzrxryrz] {self.p2_3d_base_xyzrxryrz}')
            
            ## offset
            for i in range(len(grasp_offset)):
                self.p1_3d_base_xyzrxryrz[i] += grasp_offset[i]
                self.p2_3d_base_xyzrxryrz[i] += grasp_offset[i]
            self.p1_3d_base_xyzrxryrz[4] += np.pi/6
            self.p2_3d_base_xyzrxryrz[4] += np.pi/6
            print(f'[p1_3d_base_xyzrxryrz] {self.p1_3d_base_xyzrxryrz}')
            print(f'[p2_3d_base_xyzrxryrz] {self.p2_3d_base_xyzrxryrz}')

            ## SAFTY detection BEGIN
            if thresholds is None:
                thresholds = self.grasp_thresholds
            self.start_current_monitor_thread(thresholds=thresholds)

            ## move to handle(DMP)
            print(f'Moving ...')
            tag = self.arm.move_p_dmp(self,pos=p1_3d_base_xyzrxryrz,vel=20,save_dir=f'{self.tjt_dir}/{self.action_num}/dmp/')

            ## close gripper
            print(f'Closing Gripper ...')
            if not tag:
                self.arm.control_gripper(open_value=50)
                time.sleep(2)

            ## SAFTY detection END
            self.monitor_running = False
            self.current_monitor_thread.join()
            self.vis_current_data()

            print(f'self.current_min[4-1]: {self.current_min[4-1]}')
            
            if self.this_pmt.ret == SAFTY_ISSUE or self.current_min[4-1] < -2000:
                self.this_pmt.ret = GRASP_SAFTY
                self.this_pmt.error = "GRASP_SAFTY"
            elif self.this_pmt.ret == NO_SAFTY_ISSUE:
                if tag:
                    self.this_pmt.ret = GRASP_IK_FAIL
                    self.this_pmt.error = "GRASP_IK_FAIL"
                ## grasp success if clip detecting grasping or gripper detecting grasping
                elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                    self.this_pmt.ret = GRASP_MISS
                    self.this_pmt.error = "GRASP_MISS"
                else:
                    self.this_pmt.ret = SUCCESS
                    self.this_pmt.error = "NONE"
        
        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Grasp Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def unlock(self,unlock_T=1.5,thresholds=None):
        print(f'========== Unlocking ... ==========')
        self.this_pmt.action = "UNLOCK"
        self.this_pmt.id = UNLOCK
        self.this_pmt.param = [unlock_T,0,0]

        if thresholds is None:
            thresholds = self.unlock_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(open_value=300)
        time.sleep(2)

        ## unlock
        print(f'Unlocking ...')
        # tag1,tag2 = self.arm.unlock_handle_move_p(T=unlock_T, execute_v=10,if_p=True)
        # tag = (tag1 != 0  or tag2 != 0)
        tag = self.arm.move_p(pos=self.p2_3d_base_xyzrxryrz,vel=10)
        time.sleep(1)
        
        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        ## close gripper
        print(f'Closing Gripper ...')
        if not tag:
            self.arm.control_gripper(open_value=50)
            time.sleep(3)
        
        if self.this_pmt.ret == NO_SAFTY_ISSUE:
            if tag:
                self.this_pmt.ret = UNLOCK_IK_FAIL
                self.this_pmt.error = "UNLOCK_IK_FAIL"
            elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                self.this_pmt.ret = UNLOCK_MISS
                self.this_pmt.error = "UNLOCK_MISS"
            else:
                self.this_pmt.ret = SUCCESS
                self.this_pmt.error = "NONE"
        elif self.this_pmt.ret == SAFTY_ISSUE:
            self.this_pmt.ret = UNLOCK_SAFTY
            self.this_pmt.error = "UNLOCK_SAFTY"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Unlock Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def rotate(self,rotate_T=1.5,thresholds=None):
        print(f'========== Rotating ... ==========')
        self.this_pmt.action = "ROTATE"
        self.this_pmt.id = ROTATE
        self.this_pmt.param = [rotate_T,0,0]

        if thresholds is None:
            thresholds = self.rotate_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(open_value=50)
        time.sleep(2)

        ## rotate
        print(f'Rotating ...')
        tag = self.arm.rotate_handle_move_j(T=rotate_T, execute_v=10,if_p=True)
        time.sleep(1)

        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        if self.this_pmt.ret == NO_SAFTY_ISSUE:
            if tag != 0:
                self.this_pmt.ret = ROTATE_IK_FAIL
                self.this_pmt.error = "ROTATE_IK_FAIL"
            elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                self.this_pmt.ret = ROTATE_MISS
                self.this_pmt.error = "ROTATE_MISS"
            else:
                self.this_pmt.ret = SUCCESS
                self.this_pmt.error = "NONE"
        elif self.this_pmt.ret == SAFTY_ISSUE:
            self.this_pmt.ret = ROTATE_SAFTY
            self.this_pmt.error = "ROTATE_SAFTY"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Rotate Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def open(self,open_T=2.0,thresholds=None):
        print(f'========== Opening ... ==========')
        self.this_pmt.action = "OPEN"
        self.this_pmt.id = OPEN
        self.this_pmt.param = [open_T,0,0]

        if thresholds is None:
            thresholds = self.open_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        print(f'Close Gripper ...')
        self.arm.control_gripper(open_value=50)
        time.sleep(1.5)
        print(f'opening ...')
        self.base.move_T(T=open_T)
        time.sleep(abs(open_T)+1)

        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        
        if self.this_pmt.ret == NO_SAFTY_ISSUE:
            if self.current_min[4-1] < PULL_DOOR_THRESHOLD_JOINT_4_L or self.current_max[4-1] > PUSH_DOOR_THRESHOLD_JOINT_4_H:
            # if distance < 0.5:
                self.this_pmt.ret = OPEN_FAIL
                self.this_pmt.error = "OPEN_FAIL"
                print(f'curent_min_joint_4: {self.current_min[4-1]} curent_max_joint_4: {self.current_max[4-1]}')
            elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                self.this_pmt.ret = OPEN_MISS
                self.this_pmt.error = "OPEN_MISS"
            else:
                self.this_pmt.ret = SUCCESS
                self.this_pmt.error = "NONE"
        elif self.this_pmt.ret == SAFTY_ISSUE:
            self.this_pmt.ret = OPEN_SAFTY
            self.this_pmt.error = "OPEN_SAFTY"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Open Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def home(self):
        print(f'========== Going Home ... ==========')

        self.arm.control_gripper(open_value=1000)
        time.sleep(2)
        self.base.move_T(-0.5)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_location([self.start_x,self.start_y,self.start_theta])
        self.base.move_T(0.5)
        
        self.this_pmt.action = "HOME"
        self.this_pmt.id = HOME
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Finish Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def back(self):
        ## just for debug
        print(f'========== Backing to p1_3d_base_xyzrxryrz ... ==========')
        
        self.arm.control_gripper(open_value=1000)
        time.sleep(2)
        tag2 = self.arm.move_p(pos=self.p1_3d_base_xyzrxryrz,vel=20,if_p=True)

        self.this_pmt.action = "BACK"
        self.this_pmt.id = BACK
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "BACK"
        
        self.update()

        print(f'========== Back Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def finish(self):
        print(f'========== Finishing ... ==========')

        self.arm.control_gripper(open_value=1000)
        time.sleep(2)
        self.base.move_T(-0.5)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_location([self.start_x,self.start_y,self.start_theta])
        self.base.move_T(0.5)
        
        self.this_pmt.action = "FINISH"
        self.this_pmt.id = FINISH
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "FINISH"

        self.update()
        self.save_primitives()
        
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Finish Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    def clear(self):
        print(f'========== Clearing ... ==========')

        self.action_num = 0

        self.last_pmt = _Primitive(action="START",id=START,ret=1,param=[0,0,0],error="START")
        self.this_pmt = _Primitive()

        self.primitives = {0:self.last_pmt.to_list()}
        
        self.current_max = [0]*7
        self.current_min = [0]*7
        
        if os.path.exists(self.tjt_dir):
             shutil.rmtree(self.tjt_dir)
        os.makedirs(self.tjt_dir)

        ret = 1
        error = "CLEAR"

        print(f'========== Clear Done... ==========')
        return ret,error


    def do_primitive(self,_id,_param):
        primitive_type = self.action2num(_id)
        
        ## capture
        if primitive_type == -1:
            raise ValueError("Input error: Invalid primitive type value")
        elif primitive_type == GRASP or primitive_type == PREMOVE:
            self.capture(if_d=True,vis=True,if_update=True)
        else:
            self.capture(if_d=False,vis=False,if_update=True)
        
        ## do action
        if primitive_type == PREMOVE:
            ret,error = self.premove(premove_T=_param[0])
        elif primitive_type == GRASP:
            ret,error = self.grasp(grasp_offset=_param[:3])
        elif primitive_type == ROTATE:
            ret,error = self.rotate(rotate_T=_param[0])
        elif primitive_type == UNLOCK:
            ret,error = self.unlock(unlock_T=_param[0])
        elif primitive_type == OPEN:
            ret,error = self.open(open_T=_param[0])
        elif primitive_type == HOME:
            ret,error = self.home()
        elif primitive_type == FINISH:
            ret,error = self.finish()
        elif primitive_type == BACK:
            ret,error = self.back()
        elif primitive_type == CLEAR:
            ret,error = self.clear()

        self.capture(if_d=False,vis=False,if_update=False)
        
        return ret,error

    def data_collection(self):
        num = 0
        while True:
            num += 1
            print('***************************************************************')
            user_input = input(f"[Please Input Primitive_{num}]: ")
            if user_input.lower() == 'q':
                break
            try:
                action_id, *param = [x.strip() for x in user_input.split(',')]
                param = [float(x) for x in param]
                ret, error = self.do_primitive(action_id, param)
                if error == 'FINISH':   
                    break
            except Exception as e:
                print(f"ERROR TYPE: {type(e)}: {e}")
                print("Please re-input!")
            print('***************************************************************\n\n')

    def data_collection2(self):
        ret, error = self.do_primitive('premove', [1])
        ret, error = self.do_primitive('grasp', [0,0,0])
        ret, error = self.do_primitive('open', [-3])
        

    def state_machine(self):
        state = 1
        while True:
            if state == 1:
                ret,error = self.do_primitive(_id=1,_param=1.0)
                if ret != 1:
                    pass
                else:
                    state = 2
            if state == 2:
                ret,error = self.do_primitive(_id=2,_param=[-0.04,0.03,0.01])
                if ret != 1:
                    if error == 'GRASP_IK_FAIL':
                        state = 1
                    elif error == 'GRASP_MISS':
                        state = 2
                    elif error == 'GRASP_SAFTY ':
                        state = 2
                else:
                    state = 3
            if state == 3:
                ret,error = self.do_primitive(_id=3,_param=1.8)
                if ret!= 1:
                    if error == 'UNLOCK_IK_FAIL':
                        state = 3
                    elif error == 'UNLOCK_MISS':
                        state = 3
                    elif error == 'UNLOCK_SAFTY':
                        state = 3
                else:
                    state = 4
            if state == 4:
                ret,error = self.do_primitive(_id=4,_param=2.0)
                if ret!= 1:
                    if error == 'OPEN_FAIL':
                        ret,error = self.do_primitive(_id=4,_param=-2.0)
                        state = 4

                    elif error == 'OPEN_MISS':
                        state = 4
                    elif error == 'OPEN_SAFTY':
                        state = 4
                else:
                    state = 5

            

if __name__ == '__main__':
    primitive = Primitive(root_dir='./',tjt_num=2)
    # primitive.capture()
    # primitive.premove(premove_T=-1)
    # primitive.capture(if_d=True,vis=True)
    # primitive.grasp(grasp_offset=[-0.04,0.03,0])
    primitive.capture()
    primitive.unlock(unlock_T=1.8)
    # primitive.capture()
    # primitive.open(open_T=3.0)
    # primitive.capture()
    # primitive.finish()