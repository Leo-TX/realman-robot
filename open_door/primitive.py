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
from gemini import GEMINI
from _primitive import _Primitive
from hgum import HandleGraspUnlockModel as HGUM

from utils.lib_math import *
from utils.lib_io import *
from utils.lib_rgbd import *


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
    def __init__(self,root_dir='./',tjt_num=1,cfg_path='./cfg/cfg.yaml'):
        ## get cfg
        cfg = read_yaml_file(f'{root_dir}/{cfg_path}', is_convert_dict_to_class=True)
        self.cfg = cfg

        ## init two arms
        self.arm_r = Arm.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_arm_right}')
        self.arm_l = Arm.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_arm_left}')
        
        ## init camera
        self.camera = Camera.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_cam}')
        
        ## init base
        self.base = Base.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_base}')
        
        ## init head
        self.head = Head.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_head}')
        
        ## init server
        self.server = Server.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_server}')
        
        ## init ransac
        self.ransac = RANSAC(cfg_ransac=f'{root_dir}/{cfg.cfg_ransac}',cfg_cam=f'{root_dir}/{cfg.cfg_cam}')
        
        ## init dtsam
        self.dtsam = DTSAM.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_dtsam}')

        ## init gemini
        self.gemini = GEMINI.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_gemini}')

        ## init handle_grasp_model
        self.hgum = HGUM.init_from_yaml(cfg_path=f'{root_dir}/{cfg.cfg_hgum}')

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
        
        ## current thresholds
        self.grasp_thresholds = [[self.cfg.threshold.grasp.l,self.cfg.threshold.grasp.h] for _ in range(6)]
        self.unlock_thresholds = [[self.cfg.threshold.unlock.l,self.cfg.threshold.unlock.h] for _ in range(6)]
        self.unlock_stop_thresholds_left = [[self.cfg.threshold.unlock.stop_l_left,self.cfg.threshold.unlock.stop_h_left] for _ in range(6)]
        self.unlock_stop_thresholds_right = [[self.cfg.threshold.unlock.stop_l_right,self.cfg.threshold.unlock.stop_h_right] for _ in range(6)]
        self.rotate_thresholds = [[self.cfg.threshold.rotate.l,self.cfg.threshold.rotate.h] for _ in range(6)]
        self.rotate_stop_thresholds_left = [[self.cfg.threshold.rotate.stop_l_left,self.cfg.threshold.rotate.stop_h_left] for _ in range(6)]
        self.rotate_stop_thresholds_right = [[self.cfg.threshold.rotate.stop_l_right,self.cfg.threshold.rotate.stop_h_right] for _ in range(6)]
        self.open_thresholds = [[self.cfg.threshold.open.l,self.cfg.threshold.open.h] for _ in range(6)]
        self.open_pull_thresholds_left = [[-np.inf,np.inf] for _ in range(6)]
        self.open_pull_thresholds_right = [[-np.inf,np.inf] for _ in range(6)]
        self.open_push_thresholds_left = [[-np.inf,np.inf] for _ in range(6)]
        self.open_push_thresholds_right = [[-np.inf,np.inf] for _ in range(6)]
        self.open_pull_thresholds_left[3][0] = self.cfg.threshold.open.pull_j3_l_left # pull_j3_l
        self.open_pull_thresholds_right[3][0] = self.cfg.threshold.open.pull_j3_l_right # pull_j3_l
        self.open_push_thresholds_left[3][1] = self.cfg.threshold.open.push_j3_h_left # push_j3_h
        self.open_push_thresholds_right[3][1] = self.cfg.threshold.open.push_j3_h_right # push_j3_h

        ## Primitive Types
        self.HOME = self.cfg.pmts.home
        self.PREMOVE = self.cfg.pmts.premove
        self.GRASP = self.cfg.pmts.grasp
        self.UNLOCK = self.cfg.pmts.unlock
        self.ROTATE = self.cfg.pmts.rotate
        self.OPEN = self.cfg.pmts.open
        self.START = self.cfg.pmts.start
        self.FINISH = self.cfg.pmts.finish
        self.BACK = self.cfg.pmts.back
        self.CLEAR = self.cfg.pmts.clear
        self.TELEOPERATION = self.cfg.pmts.teleoperation
        
        ## Error Types
        self.SUCCESS = self.cfg.errors.success
        self.SAFETY_ISSUE = self.cfg.errors.current.safety_issue
        self.EVENT_DETECTED = self.cfg.errors.current.event_detected
        self.NO_ISSUE = self.cfg.errors.current.no_issue
        self.GRASP_SAFETY = self.cfg.errors.grasp.grasp_safety
        self.GRASP_NO_HANDLE = self.cfg.errors.grasp.grasp_no_handle
        self.GRASP_IK_FAIL = self.cfg.errors.grasp.grasp_ik_fail
        self.GRASP_MISS = self.cfg.errors.grasp.grasp_miss
        self.ROTATE_SAFETY = self.cfg.errors.rotate.rotate_safety
        self.ROTATE_MISS = self.cfg.errors.rotate.rotate_miss
        self.ROTATE_IK_FAIL = self.cfg.errors.rotate.rotate_ik_fail
        self.UNLOCK_SAFETY = self.cfg.errors.unlock.unlock_safety
        self.UNLOCK_MISS = self.cfg.errors.unlock.unlock_miss
        self.UNLOCK_IK_FAIL = self.cfg.errors.unlock.unlock_ik_fail
        self.OPEN_SAFETY = self.cfg.errors.open.open_safety
        self.OPEN_MISS = self.cfg.errors.open.open_miss
        self.OPEN_FAIL = self.cfg.errors.open.open_fail

        ## init 
        self.last_pmt = _Primitive(action="START",id=self.START,ret=1,param=[0,0,0],error="START")
        self.this_pmt = _Primitive(action="START",id=self.START,ret=1,param=[0,0,0],error="START")
        self.primitives = {0:self.last_pmt.to_list()}

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
        if action == "premove":
            return self.PREMOVE
        elif action == "grasp":
            return self.GRASP
        elif action == "unlock":
            return self.UNLOCK
        elif action == "rotate":
            return self.ROTATE
        elif action == "open":
            return self.OPEN
        elif action == "home":
            return self.HOME
        elif action == "finish":
            return self.FINISH
        elif action == "back":
            return self.BACK
        elif action == "clear":
            return self.CLEAR
        elif action == "teleoperation" or action == "tele":
            return self.TELEOPERATION
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
                vis_rgbd(d_img_path=self.d_img_path,rgb_img_path=self.rgb_img_path,save_path=f'{self.tjt_dir}/{self.action_num}/vis_rgbd.png')
                vis_d(d_img_path=self.d_img_path,save_path=f'{self.tjt_dir}/{self.action_num}/vis_d.png')
            return rgb_img,d_img 
        else:
            if if_update:
                rgb_img = self.camera.capture_rgb(rgb_save_path=self.rgb_img_path)
            else:
                rgb_img = self.camera.capture_rgb(f'{self.tjt_dir}/temp.png')
            return rgb_img
        # print(f'========== Image Captured ==========')

    def start_current_monitor_thread(self,thresholds_safety,thresholds_event=None,if_event_stop=True):
        print(f'thresholds_safety: {thresholds_safety}')
        print(f'thresholds_event: {thresholds_event}')
        self.current_data = {i: [] for i in range(7)}
        self.current_max = [0]*7
        self.current_min = [0]*7
        self.current_data_start_time = time.time()
        self.monitor_running = True
        self.current_monitor_thread = threading.Thread(target=self.current_monitor_loop, args=(thresholds_safety,thresholds_event,if_event_stop))
        self.current_monitor_thread.daemon = True
        self.current_monitor_thread.start()

    def current_monitor_loop(self,thresholds_safety,thresholds_event,if_event_stop):
        while self.monitor_running:
            current_check_result,current = self.check_current_safety(thresholds_safety,thresholds_event)
            if current_check_result == -1:
                print(f"!!! SAFETY ISSUE !!!")
                self.this_pmt.ret = self.SAFETY_ISSUE
                self.this_pmt.error = 'SAFETY_ISSUE'
                print(f'[Now Current]: {current}')
                self.arm.move_stop(if_p=True)
                move_stop
                break
            elif current_check_result == 0:
                print(f"!!! Event Detected !!!")
                self.this_pmt.ret = self.EVENT_DETECTED
                self.this_pmt.error = 'EVENT_DETECTED'
                print(f'[Now Current]: {current}')
                self.action_T = time.time() - self.current_data_start_time
                if if_event_stop:
                    self.arm.move_stop(if_p=True)
                    self.base.move_stop(if_p=True)
                break
            elif current_check_result == 1:
                self.this_pmt.ret = self.NO_ISSUE
                self.this_pmt.error = 'NO_ISSUE'
            time.sleep(0.1)

    def check_current_safety(self,thresholds_safety,thresholds_event):
        current = self.arm.get_c()
        for i in range(7):
            self.current_data[i].append(current[i])
            self.current_max[i] = max(self.current_max[i],current[i])
            self.current_min[i] = min(self.current_min[i],current[i])
        # safety issue
        for i, (min_current, max_current) in enumerate(thresholds_safety):
            if current[i] < min_current or current[i] > max_current:
                return -1,current
        # event detected
        if thresholds_event:
            for i, (min_current, max_current) in enumerate(thresholds_event):
                if current[i] < min_current or current[i] > max_current:
                    return 0,current
        # no issue
        return 1,current

    def vis_current_data(self,save_path=None,show=False):
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
    def premove(self):
        print(f'========== Premoving ... ==========')
        self.this_pmt.action = "PREMOVE"
        self.this_pmt.id = self.PREMOVE
        self.this_pmt.param = [0,0,0]

        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        if not os.path.exists(os.path.dirname(rgb_img_path)):
            os.makedirs(os.path.dirname(rgb_img_path))
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path

        self.normal,self.weights,self._3d_center,self._2d_center,self.mask_color = self.ransac.get_normal_server(rgb_img_path,d_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
        print(f'[RANSAC Result] normal: {self.normal} weights: {self.weights}')

        self.move_T = self.base.move_to_door(self.weights,self.cfg.premove.offset_in_front,self.cfg.premove.d2t_coefficient)
        time.sleep(2)

        self.this_pmt.ret = self.SUCCESS
        self.this_pmt.error = "NONE"

        self.this_pmt.param = [self.move_T,0,0]

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Premove Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def grasp(self,grasp_param):
        print('========== Grasping... ==========')
        
        self.this_pmt.action = "GRASP"
        self.this_pmt.id = self.GRASP
        self.this_pmt.param = grasp_param
        self.dx,self.dy,self.R = grasp_param

        ## os
        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        if not os.path.exists(os.path.dirname(rgb_img_path)):
            os.makedirs(os.path.dirname(rgb_img_path))
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path
        
        ## dtsam
        print('DTSAM ...')
        self.x1_2d,self.y1_2d,self.orientation,self.w,self.h,self.box = self.dtsam.get_xy_server(rgb_img_path,self.server,self.remote_python_path,self.remote_root_dir,self.remote_img_dir)
        if self.w == 0 and self.h == 0:
            self.this_pmt.ret = self.GRASP_NO_HANDLE
            self.this_pmt.error = "GRASP_NO_HANDLE"
            print(f'[DTSAM Result] NO handle detections!!!')
        else:
            ##　grasp point 2d offset(dx,dy)
            self.x1_2d += self.dx
            self.y1_2d += self.dy
            
            ## rotate point
            self.x2_2d,self.y2_2d,self.Ox,self.Oy = rotate_point(self.x1_2d,self.y1_2d,R=self.R,orientation=self.orientation,angle=90)
            print(f'[p1_2d] x1_2d: {self.x1_2d}, y1_2d: {self.y1_2d}')
            print(f'[p2_2d] x2_2d: {self.x2_2d}, y2_2d: {self.y2_2d}')
            print(f'[center] Ox: {self.Ox}, Oy: {self.Oy}')
            
            ## vis
            vis_grasp(rgb_img_path,self.dx,self.dy,self.x1_2d,self.y1_2d,self.x2_2d,self.y2_2d,self.Ox,self.Oy,self.R,self.orientation,angle=90,save_path=rgb_img_path.replace('rgb','vis_grasp'),show=False)

            ## determin which arm
            if self.x1_2d < self.camera.width / 2:
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
            print(f'[p1_3d_cam_xyzrxryrz] {self.p1_3d_cam_xyzrxryrz}')
            print(f'[p2_3d_cam_xyzrxryrz] {self.p2_3d_cam_xyzrxryrz}')
            
            self.p1_3d_base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(self.p1_3d_cam_xyzrxryrz)
            self.p2_3d_base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(self.p2_3d_cam_xyzrxryrz)
            print(f'[p1_3d_base_xyzrxryrz] {self.p1_3d_base_xyzrxryrz}')
            print(f'[p2_3d_base_xyzrxryrz] {self.p2_3d_base_xyzrxryrz}')

            ## p1 offset and p2 offset
            def p1_offset(xyzrxryrz,r_l,orientation):
                _x,_y,_z,_rx,_ry,_rz = xyzrxryrz
                if r_l == 'right':
                    x = _x + self.cfg.grasp.p1_depth_offset
                    y = _y
                    z = _z
                    rx = _rx
                    ry = _ry+np.pi/6-np.pi/2
                    rz = _rz
                    if orientation == 'vertical':
                        ry += np.pi/2
                elif r_l == 'left':
                    x = _x
                    y = _y - self.cfg.grasp.p1_depth_offset
                    z = _z
                    rx = -1*_ry
                    ry = _rz+np.pi*2/3+np.pi/18
                    rz = -1*_rx-np.pi
                    if orientation == 'vertical':
                        rx,ry,rz = [1.0540000200271606, 0.9549999833106995, -0.6980000138282776]

                return [x,y,z,rx,ry,rz]
            
            def p2_offset(xyzrxryrz,r_l,orientation,R):
                _x,_y,_z,_rx,_ry,_rz = xyzrxryrz
                if r_l == 'right':
                    x = _x + self.cfg.grasp.p2_depth_offset
                    y = _y
                    z = _z
                    rx = _rx
                    ry = _ry
                    rz = _rz
                    if R > 0:
                        ry -= np.pi/2
                    else:
                        ry += np.pi/2
                
                elif r_l == 'left':
                    x = _x
                    y = _y - self.cfg.grasp.p2_depth_offset
                    z = _z
                    rx = _rx
                    ry = _ry
                    rz = _rz
                    if orientation == 'horizontal':
                        if R > 0:
                            rx,ry,rz = [-0.7269999980926514, -0.7990000247955322, 2.127000093460083]
                        else:
                            rx,ry,rz = [0.7269999980926514, 0.7990000247955322, -1.0130000114440918]
                    elif orientation == 'vertical':
                        if R > 0:
                            rx,ry,rz = [1.2350000143051147, -0.5249999761581421, -0.09000000357627869]
                        else:
                            rx,ry,rz = [-1.2350000143051147, 0.5249999761581421, 3.049999952316284]

                return [x,y,z,rx,ry,rz]

            self.p1_3d_base_xyzrxryrz = p1_offset(self.p1_3d_base_xyzrxryrz,self.r_l,self.orientation)
            self.p2_3d_base_xyzrxryrz[3:6] = self.p1_3d_base_xyzrxryrz[3:6]
            self.p2_3d_base_xyzrxryrz = p2_offset(self.p2_3d_base_xyzrxryrz,self.r_l,self.orientation,self.R)
            print(f'[p1_3d_base_xyzrxryrz] {self.p1_3d_base_xyzrxryrz}')
            print(f'[p2_3d_base_xyzrxryrz] {self.p2_3d_base_xyzrxryrz}')

            ## Current Detection Begin
            self.start_current_monitor_thread(thresholds_safety=self.grasp_thresholds)

            ## move to handle(DMP)
            print(f'Moving ...')
            tag = self.arm.move_p_dmp(pos=self.p1_3d_base_xyzrxryrz,save_dir=f'{self.tjt_dir}/{self.action_num}/dmp/',if_p=True,if_planb=True)
            
            ## close gripper
            print(f'Closing Gripper ...')
            if not tag:
                self.arm.control_gripper(self.cfg.grasp.gripper_value)
                time.sleep(2)

            ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
            self.monitor_running = False
            self.current_monitor_thread.join()
            self.vis_current_data()
            
            ## update
            if self.this_pmt.ret == self.SAFETY_ISSUE:
                self.this_pmt.ret = self.GRASP_SAFETY
                self.this_pmt.error = "GRASP_SAFETY"
            elif self.this_pmt.ret == self.NO_ISSUE or self.this_pmt.ret == self.EVENT_DETECTED:
                if tag:
                    self.this_pmt.ret = self.GRASP_IK_FAIL
                    self.this_pmt.error = "GRASP_IK_FAIL"
                elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                    self.this_pmt.ret = self.GRASP_MISS
                    self.this_pmt.error = "GRASP_MISS"
                else:
                    self.this_pmt.ret = self.SUCCESS
                    self.this_pmt.error = "NONE"
        
        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Grasp Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def unlock(self):
        print(f'========== Unlocking ... ==========')
        
        self.this_pmt.action = "UNLOCK"
        self.this_pmt.id = self.UNLOCK
        self.this_pmt.param = [0,0,0]

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(self.cfg.unlock.gripper_value_before)
        time.sleep(2)

        ## Current Detection Begin
        if self.r_l == 'right':
            self.start_current_monitor_thread(thresholds_safety=self.unlock_thresholds,thresholds_event=self.unlock_stop_thresholds_right,if_event_stop=True)
        elif self.r_l == 'left':
            self.start_current_monitor_thread(thresholds_safety=self.unlock_thresholds,thresholds_event=self.unlock_stop_thresholds_left,if_event_stop=True)

        ## unlock
        print(f'Unlocking ...')
        tag = self.arm.move_p(pos=self.p2_3d_base_xyzrxryrz,vel=self.cfg.unlock.unlock_v,if_p=True)
        time.sleep(2)
        
        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        ## close gripper
        print(f'Closing Gripper ...')
        if not self.this_pmt.ret == self.SAFETY_ISSUE:
            self.arm.control_gripper(self.cfg.unlock.gripper_value_after)
            time.sleep(2)
        
        ## update
        if self.this_pmt.ret == self.NO_ISSUE:
            if tag:
                self.this_pmt.ret = self.UNLOCK_IK_FAIL
                self.this_pmt.error = "UNLOCK_IK_FAIL"
            elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                self.this_pmt.ret = self.UNLOCK_MISS
                self.this_pmt.error = "UNLOCK_MISS"
            else:
                self.this_pmt.ret = self.SUCCESS
                self.this_pmt.error = "NONE"
        elif self.this_pmt.ret == self.SAFETY_ISSUE:
            self.this_pmt.ret = self.UNLOCK_SAFETY
            self.this_pmt.error = "UNLOCK_SAFETY"

        unlock_T = self.action_T * np.sign(self.R)
        self.this_pmt.param = [unlock_T,0,0]

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Unlock Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def rotate(self):
        print(f'========== Rotating ... ==========')
        
        self.this_pmt.action = "ROTATE"
        self.this_pmt.id = self.ROTATE
        self.this_pmt.param = [0,0,0]

        ## Current Detection Begin
        if self.r_l == 'right':
            self.start_current_monitor_thread(thresholds_safety=self.rotate_thresholds,thresholds_event=self.rotate_stop_thresholds_right,if_event_stop=True)
        elif self.r_l == 'left':
            self.start_current_monitor_thread(thresholds_safety=self.rotate_thresholds,thresholds_event=self.rotate_stop_thresholds_left,if_event_stop=True)

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(self.cfg.rotate.gripper_value)
        time.sleep(2)

        ## rotate
        print(f'Rotating ...')
        tag = self.arm.move_j(joint=self.arm.get_j()-180,vel=self.cfg.rotate.rotate_v)
        time.sleep(2)

        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        ## update
        if self.this_pmt.ret == self.NO_ISSUE:
            if tag != 0:
                self.this_pmt.ret = self.ROTATE_IK_FAIL
                self.this_pmt.error = "ROTATE_IK_FAIL"
            elif self.arm.get_gripper_grasp_return(if_p=True) != 2:
                self.this_pmt.ret = self.ROTATE_MISS
                self.this_pmt.error = "ROTATE_MISS"
            else:
                self.this_pmt.ret = self.SUCCESS
                self.this_pmt.error = "NONE"
        elif self.this_pmt.ret == self.SAFETY_ISSUE:
            self.this_pmt.ret = self.ROTATE_SAFETY
            self.this_pmt.error = "ROTATE_SAFETY"

        rotate_T = self.action_T
        self.this_pmt.param = [rotate_T,0,0]

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Rotate Done ==========')
        return self.this_pmt.ret,self.this_pmt.error
    
    @time_it
    def open(self):
        print(f'========== Opening ... ==========')
        
        self.this_pmt.action = "OPEN"
        self.this_pmt.id = self.OPEN
        self.this_pmt.param = [0,0,0]

        if self.r_l == 'right':
            open_push_thresholds = self.open_push_thresholds_right
            open_pull_thresholds = self.open_pull_thresholds_right
            direction = -1
        elif self.r_l == 'left':
            open_push_thresholds = self.open_push_thresholds_left
            open_pull_thresholds = self.open_pull_thresholds_left
            direction = 1

        self.ps_pl = 0

        ## Current Detection Begin
        self.start_current_monitor_thread(thresholds_safety=self.open_thresholds,thresholds_event=open_pull_thresholds,if_event_stop=True)

        ## close gripper
        print(f'Close Gripper ...')
        self.arm.control_gripper(self.cfg.open.gripper_value_pull)
        time.sleep(2)
    
        ## open (pull)
        print(f'opening (pull)...')
        self.base.move_open_door(self.cfg.open.T,-self.cfg.open.linear_velocity,-self.cfg.open.angular_velocity*direction)

        ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        if self.this_pmt.ret != self.SAFETY_ISSUE:
            ## handle slip
            if self.arm.get_gripper_grasp_return(if_p=True) != 2:
                    self.this_pmt.ret = self.OPEN_MISS
                    self.this_pmt.error = "OPEN_MISS"
            else:
                ## pull failed
                if self.this_pmt.ret == self.EVENT_DETECTED:
                    # ## Current Detection Begin
                    # self.start_current_monitor_thread(thresholds_safety=self.open_thresholds,thresholds_event=open_push_thresholds,if_event_stop=False)
                    
                    ## close gripper
                    print(f'Close Gripper ...')
                    self.arm.control_gripper(self.cfg.open.gripper_value_push)
                    time.sleep(2)

                    ## open (push)
                    print(f'opening (push)...')
                    self.base.move_open_door(self.cfg.open.T,self.cfg.open.linear_velocity,self.cfg.open.angular_velocity*direction)

                    # ## Current Detection End (1.[safety issue] or 2.[event detected] or 3.[code runs to this line])
                    # self.monitor_running = False
                    # self.current_monitor_thread.join()
                    # self.vis_current_data()

                    # if self.this_pmt.ret == self.NO_ISSUE:
                    self.ps_pl = -1 # 'push'
                    print(f'pushing successed ...')
                    self.this_pmt.ret = self.SUCCESS
                    self.this_pmt.error = "NONE"

                    # elif self.this_pmt.ret == self.SAFETY_ISSUE:
                    #     self.this_pmt.ret = self.OPEN_SAFETY
                    #     self.this_pmt.error = "OPEN_SAFETY"
                    
                ## pull successed
                else:
                    self.ps_pl = 1 # 'pull'
                    print(f'pulling successed ...')
                    self.this_pmt.ret = self.SUCCESS
                    self.this_pmt.error = "NONE"

        else:
            self.this_pmt.ret = self.OPEN_SAFETY
            self.this_pmt.error = "OPEN_SAFETY"

        self.this_pmt.param = [self.ps_pl,0,0]

        self.update()
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Open Done ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def teleoperation(self,interval=0.1):
        print(f'========== Teleoperation ... ==========')
       
        while True:
            try: 
                char = getch(if_p=True)
                if char in ['w','a','s','d','H','P','K','M']:
                    self.base.move_char(char)
                elif char == '0':
                    self.arm.go_home(block=False)
                elif char == '8':
                    self.arm.control_gripper(self.cfg.back.gripper_value)
                    time.sleep(2)
                    self.arm.move_p(pos=self.p1_3d_base_xyzrxryrz,if_p=True,block=False)
                elif char == 'q':
                    break
                time.sleep(interval)
            except KeyboardInterrupt:
                break

        self.this_pmt.action = "TELEOPERATION"
        self.this_pmt.id = self.TELEOPERATION
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Finish Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def home(self):
        print(f'========== Going Home ... ==========')

        self.arm.control_gripper(self.cfg.home.gripper_value)
        time.sleep(2)
        self.base.move_T(-self.cfg.home.move_T)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_T(self.cfg.home.move_T)
        time.sleep(1)

        self.this_pmt.action = "HOME"
        self.this_pmt.id = self.HOME
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "None"

        self.update()
        
        print(f'[Primitive INFO] ret: {self.this_pmt.ret}, error: {self.this_pmt.error}')
        print(f'========== Finish Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def back(self):
        ## back to grasping state
        print(f'========== Backing to p1_3d_base_xyzrxryrz ... ==========')
        
        self.arm.control_gripper(self.cfg.back.gripper_value)
        time.sleep(2)
        tag = self.arm.move_p(pos=self.p1_3d_base_xyzrxryrz,if_p=True)

        self.this_pmt.action = "BACK"
        self.this_pmt.id = self.BACK
        self.this_pmt.ret = 1
        self.this_pmt.param = [0,0,0]
        self.this_pmt.error = "BACK"
        
        self.update()

        print(f'========== Back Done... ==========')
        return self.this_pmt.ret,self.this_pmt.error

    @time_it
    def finish(self):
        print(f'========== Finishing ... ==========')

        self.arm.control_gripper(self.cfg.finish.gripper_value)
        time.sleep(2)
        self.base.move_T(-self.cfg.finish.move_T)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_location([self.base.start_x,self.base.start_y,self.base.start_theta])
        time.sleep(1)
        self.base.move_T(self.cfg.finish.move_T)
        time.sleep(1)
        
        self.this_pmt.action = "FINISH"
        self.this_pmt.id = self.FINISH
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

        self.last_pmt = _Primitive(action="START",id=self.START,ret=1,param=[0,0,0],error="START")
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

    def do_primitive(self,_id,_param=None):
        primitive_type = self.action2num(_id)
        
        ## capture
        if primitive_type == -1:
            raise ValueError("Input error: Invalid primitive type value")
        elif primitive_type == self.GRASP or primitive_type == self.PREMOVE:
            self.capture(if_d=True,vis=True,if_update=True)
        else:
            self.capture(if_d=False,vis=False,if_update=True)
        
        ## do action
        if primitive_type == self.PREMOVE:
            ret,error = self.premove()
        elif primitive_type == self.GRASP:
            ret,error = self.grasp(grasp_param=_param[:3])
        elif primitive_type == self.ROTATE:
            ret,error = self.rotate()
        elif primitive_type == self.UNLOCK:
            ret,error = self.unlock()
        elif primitive_type == self.OPEN:
            ret,error = self.open()
        elif primitive_type == self.HOME:
            ret,error = self.home()
        elif primitive_type == self.FINISH:
            ret,error = self.finish()
        elif primitive_type == self.BACK:
            ret,error = self.back()
        elif primitive_type == self.CLEAR:
            ret,error = self.clear()
        elif primitive_type == self.TELEOPERATION:
            ret,error = self.teleoperation()

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
                if param:
                    ret, error = self.do_primitive(action_id, param)
                else:
                    ret, error = self.do_primitive(action_id)
                if error == 'FINISH':   
                    break
            except Exception as e:
                print(f"ERROR TYPE: {type(e)}: {e}")
                print("Please re-input!")
            print('***************************************************************\n\n')

    def open_loop(self):
        ret, error = self.do_primitive('premove')
        ret, error = self.do_primitive('grasp', [0,0,0])
        ret, error = self.do_primitive('unlock')
        ret, error = self.do_primitive('open')
    
    def hl_LLM(self):
        prompt = """"""
        img_path = ''
        response = self.gemini.text_to_text(prompt)
        next_id = int(response)

    def hl_VLM(self):
        prompt = """"""
        img_path = ''
        response = self.gemini.text_img_to_text(prompt,img)
        next_id = int(response)

    def hl_MLP(self):
        pass

    def hl_SM(self):
        pass

    def ll_YOLO(self):
        pass

    def ll_VLM(self):
        pass
        
    def close_loop_state_machine(self):
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
                    elif error == 'GRASP_SAFETY ':
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
                    elif error == 'UNLOCK_SAFETY':
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
                    elif error == 'OPEN_SAFETY':
                        state = 4
                else:
                    state = 5

if __name__ == '__main__':
    primitive = Primitive(root_dir='./',tjt_num=2)