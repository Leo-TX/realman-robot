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
import clip
import torch
from PIL import Image
import threading
from matplotlib import pyplot as plt

from arm import Arm
from base import Base
from camera import Camera
from head import Head
from dtsam import DTSAM
from server import Server
from ransac import RANSAC
from dmp import DMP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Primitive Types
# DUAL = 0 # [0/1,0,0] # 0 for right 1 for left
FINISH = 10086
PREMOVE = 1 # [T,0,0]
GRASP = 2 # [dx,dy,dz]
UNLOCK = 3 # [T,0,0]
ROTATE = 4 # [T,0,0]
OPEN = 5 # [T,0,0]

## Error Types
SUCCESS = 1

SAFTY_ISSUE = 2131
NO_SAFTY_ISSUE = 3121

PREMOVE_TOO_CLOSE = -1 # CLIP: "There is/isn't a handle."

GRASP_IK_FAIL = -1 # IK Error(can't reach)
GRASP_MISS = -1 # CLIP: "The gripper is/isn't grasping the handle."
GRASP_SAFTY = 0 # Current Detection

ROTATE_MISS = -1 # CLIP: "The gripper is/isn't grasping the handle./The gripper API?"
ROTATE_SAFTY = 0 # Current Detection(rotating too much)
ROTATE_IK_FAIL = -1

UNLOCK_MISS = -1 # CLIP: "The gripper is/isn't grasping the handle./The gripper API?"
UNLOCK_SAFTY = 0 # Current Detection(unlocking too much)
UNLOCK_IK_FAIL = -1

OPEN_MISS = -1 # CLIP
OPEN_SAFTY = -1
OPEN_FAIL = -1 # The location of the base doesn't change too much

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
    def __init__(self,root_dir='./',tjt_num=1):
        self.primitives = []
        self.root_dir = root_dir

        # trajectory
        self.tjt_num = tjt_num
        self.tjt_dir = f'{self.root_dir}/data/trajectory{self.tjt_num}/'
        if os.path.exists(self.tjt_dir):
             shutil.rmtree(self.tjt_dir)
        os.makedirs(self.tjt_dir)

        # cfg
        self.cfg_dir = f'{self.root_dir}/cfg/'
        self.cam_params_path = f'{self.cfg_dir}/cam_params.json'
        self.cam2base_H_path = f'{self.cfg_dir}/cam2base_H.csv'
        
        self.action_num = 0
        self.this_id = 0
        self.this_ret = 0
        self.this_param = [0,0,0]
        self.this_error = ''
        self.last_id = 0
        self.last_ret = 0
        self.last_param = [0,0,0]
        self.last_error = ''
        
        self.clip_model, self.preprocess = None, None

        self.grasp_thresholds = [[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000]]
        self.unlock_thresholds = [[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000]] # 10000
        self.rotate_thresholds = [[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000]]
        self.open_thresholds = [[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000],[-100000, 100000]]
        
        self.connect_robot()
    
    def connect_robot(self):
        ## init camera
        self.camera = Camera(cam_params_path=self.cam_params_path,fps=30)

        ## init arm
        self.arm = Arm('192.168.10.19',8080,cam2base_H_path=self.cam2base_H_path,if_gripper=True,if_monitor=False,tool_frame='dh3')# 18 for left 19 for right
        self.arm.control_gripper(open_value=1000)
        self.arm.go_home()

        ## init base
        self.base = Base(host_ip='192.168.10.10',host_port=31001,linear_velocity=0.2,angular_velocity=1.0)
        self.start_x,self.start_y,self.start_theta = self.base.get_location()

        ## init head
        self.head = Head(port='COM3',baudrate=9600)
        self.head.servo_move(1000, 1, 400)
        self.head.servo_move(1000, 2, 500)

        ## init server
        self.server = Server(hostname='130.126.136.95',username='zhi',password='yourpassword',if_stfp=True)

    def disconnect_robot(self):
        print('========== Disconnecting... ==========')
        camera = camera.disconnect()
        arm.disconnect()
        base.disconnect()
        head.disconnect()
        server.disconnect()
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
        elif action == "finish":
            return FINISH
        else:
            return -1

    def update(self,save_path=None):
        if save_path is None:
            save_path=f'{self.tjt_dir}/{self.action_num}.json'
        data = {"last_id": self.last_id,
                "last_ret": self.last_ret,
                "last_param": self.last_param,
                "last_error": self.last_error,
                "this_id": self.this_id,
                "this_ret": self.this_ret,
                "this_param": self.this_param,
                "this_error": self.this_error
                }
        with open(save_path,'w') as json_file:
            json.dump(data,json_file,indent=4)
        self.last_id = self.this_id
        self.last_ret = self.this_ret
        self.last_param = self.this_param
        self.last_error = self.this_error

        self.primitives.append([self.this_id,self.this_ret,self.this_param,self.this_error])


    def CLIP(self,rgb_img, text_prompt, model_name = "ViT-B/32", if_p = False):
        if self.clip_model is None or self.preprocess is None:
            self.clip_model, self.preprocess = clip.load(model_name, device=DEVICE) # Load CLIP model
        if isinstance(rgb_img,str):
            rgb_img_input = self.preprocess(Image.open(rgb_img)).unsqueeze(0).to(DEVICE)# get rgb_img_input
        else:
            rgb_img_input = self.preprocess(rgb_img).unsqueeze(0).to(DEVICE)# get rgb_img_input
        text_input = clip.tokenize(text_prompt).to(DEVICE)# get text_input
        with torch.no_grad():
            image_features = self.clip_model.encode_image(rgb_img_input)
            text_features = self.clip_model.encode_text(text_input)
            logits_per_image, logits_per_text = self.clip_model(rgb_img_input, text_input)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            if if_p:
                print("Label probs:", probs)  # prints: [[0.9927937  0.00421068]]
        return probs

    def CLIP_detection(self,rgb_img=None,text_prompt=["door that is closed", "door that is open"],if_p=False):
        if rgb_img is None:
            rgb_img = Image.fromarray(self.camera.capture_rgb())
        probs = self.CLIP(rgb_img, text_prompt) # using CLIP
        result = 1 if probs[0][1] > probs[0][0] else 0 # Reward is 1 if closer to 'open door' prompt, 0 otherwise
        if if_p:
            # print("result:", result)  # prints: 1/0
            print(f'[CLIP INFO] Result: {text_prompt[result]},\tLabel probs: {probs}') # door is open/closed
        return result

    # @time_it
    def capture(self,if_d=False,vis=False):
        # print('========== Image Capturing ... ==========')
        self.action_num += 1
        self.rgb_img_path = f'{self.tjt_dir}/{self.action_num}.png'
        if if_d:
            self.d_img_path = f'{self.tjt_dir}/{self.action_num}/d.png'
            if not os.path.exists(os.path.dirname(self.d_img_path)):
                os.makedirs(os.path.dirname(self.d_img_path))
            self.camera.capture_rgbd(rgb_save_path=self.rgb_img_path,d_save_path=self.d_img_path)
            if vis:
                self.camera.vis_rgbd(d_img_path=self.d_img_path,rgb_img_path=self.rgb_img_path,save_path=f'{self.tjt_dir}/{self.action_num}/rgbd_vis.png')
                self.camera.vis_d(d_img_path=self.d_img_path,save_path=f'{self.tjt_dir}/{self.action_num}/d_vis.png')
        else:
            self.camera.capture_rgb(rgb_save_path=self.rgb_img_path)
        # print(f'========== Image Captured ==========')
        return 1

    def start_current_monitor_thread(self, thresholds):
        self.current_data = {i: [] for i in range(7)}
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
                self.this_ret = SAFTY_ISSUE
                self.this_error = 'SAFTY_ISSUE'
                self.arm.move_stop(if_p=True)
                break
            else:
                self.this_ret = NO_SAFTY_ISSUE
                self.this_error = 'NO_SAFTY_ISSUE'
            time.sleep(0.1)

    def check_current_safety(self, thresholds):
        current = self.arm.get_c()
        for i in range(7):
            self.current_data[i].append(current[i])
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
        self.this_id = PREMOVE
        self.this_param = [premove_T,0,0]

        self.base.move_T(T=premove_T)
        time.sleep(2)
        if self.CLIP_detection(rgb_img=Image.fromarray(self.camera.capture_rgb()), text_prompt=['door with a handle','door without a handle'],if_p=True) == 1:
            self.this_ret = PREMOVE_TOO_CLOSE
            self.this_error = "PREMOVE_TOO_CLOSE"
        else:
            self.this_ret = SUCCESS
            self.this_error = "NONE"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_ret}, error: {self.this_error}')
        print(f'========== Premove Done ==========')
        return self.this_ret,self.this_error
    
    @time_it
    def grasp(self,grasp_offset=[-0.04,0.03,0],thresholds=None):
        print('========== Grasping... ==========')
        self.this_id = GRASP
        self.this_param = grasp_offset
        if thresholds is None:
            thresholds = self.grasp_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        # remote
        device = 'cuda:2'
        remote_python_path = f'/media/datadisk10tb/leo/anaconda3/envs/rm/bin/python'
        remote_root_dir = f'/media/datadisk10tb/leo/projects/realman-robot/open_door/'
        remote_img_dir = f'{remote_root_dir}/data/trajectory{self.tjt_num}/{self.tjt_num}/'
        rgb_img_path = f'{self.tjt_dir}/{self.action_num}/rgb.png'
        shutil.copy2(self.rgb_img_path, rgb_img_path)
        d_img_path = self.d_img_path
        ransac_cfg_path = f'{self.cfg_dir}/ransac_cfg.yaml'
        dmp_refer_tjt_path = f'{self.cfg_dir}/refer_tjt.csv'
        refer_tjt_path = f'{self.tjt_dir}/{self.action_num}/dmp/refer_tjt.csv'
        if not os.path.exists(os.path.dirname(refer_tjt_path)):
            os.makedirs(os.path.dirname(refer_tjt_path))
        shutil.copy2(dmp_refer_tjt_path, refer_tjt_path)

        ## dtsam
        print('DTSAM ...')
        self.dtsam = DTSAM(img_path=rgb_img_path,classes='handle',device=device,threshold=0.3)
        x1_2d,y1_2d,orientation,w,h = self.dtsam.get_xy_paramiko(self.server,remote_python_path,remote_root_dir,remote_img_dir)
        y1_2d -= 5 # for avoiding depth value error(zero)
        print(f'DTSAM Result: x1_2d: {x1_2d}, y1_2d: {y1_2d}, orientation: {orientation}, w: {w}, h: {h}')

        ## xy2xyz
        x1_3d,y1_3d,z1_3d,average_depth = self.camera.xy2xyz(x1_2d,y1_2d,d_img=d_img_path) # or d_img_path
        print(f'xy2xyz Result: x1_3d: {x1_3d}, y1_3d: {y1_3d}, z1_3d: {z1_3d}, average_depth: {average_depth}')

        ## ransac
        print('RANSAC ...')

        self.ransac = RANSAC(rgb_img_path=rgb_img_path,d_img_path=d_img_path,config_file_path=ransac_cfg_path,camera_info_file_path=self.cam_params_path,vis=False)
        normal = self.ransac.get_normal_paramiko(self.server,remote_python_path,remote_root_dir,remote_img_dir)
        print(f'RANSAC Result: normal: {normal}')
        
        ## normal2rxryrz
        rx,ry,rz = self.camera.normal2rxryrz(normal)
        print(f'normal2rxryrz Result: rx: {rx} ry: {ry} rz: {rz}')

        ## target2cam_xyzrxryrz 2 target2base_xyzrxryrz
        target2cam_xyzrxryrz = [x1_3d,y1_3d,z1_3d,rx,ry,rz]
        target2base_xyzrxryrz = self.arm.target2cam_xyzrpy_to_target2base_xyzrpy(target2cam_xyzrxryrz,if_gripper=False)
        if orientation == 'horizontal':
            target2base_xyzrxryrz[4] -= np.pi/2
        print(f'target2base_xyzrxryrz: {target2base_xyzrxryrz}')
        
        ## offset
        goal_pos = target2base_xyzrxryrz
        for i in range(len(grasp_offset)):
            goal_pos[i] += grasp_offset[i]
        goal_pos[4] += np.pi/6
        print(f'goal_pos: {goal_pos}')

        ## dmp
        print(f'DMP ...')
        self.dmp = DMP(refer_tjt_path)
        new_tjt = self.dmp.gen_new_tjt(initial_pos=self.arm.get_p(),goal_pos=goal_pos,show=False)
        middle_pose = self.dmp.get_middle_pose(tjt=new_tjt,num=100)
        print(f'DMP Result: middle_pose: {middle_pose}')
        
        ## move to handle
        print(f'Moving ...')
        tag1 = self.arm.move_p(pos=middle_pose,vel=10,if_p=True)
        tag2 = self.arm.move_p(pos=goal_pos,vel=10,if_p=True)
        
        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        if self.this_ret == NO_SAFTY_ISSUE:
            if tag1 !=0 or tag2 != 0:
                self.this_ret = GRASP_IK_FAIL
                self.this_error = "GRASP_IK_FAIL"
            elif self.CLIP_detection(rgb_img=Image.fromarray(self.camera.capture_rgb()), text_prompt=['gripper on the handle','gripper not on the handle'],if_p=True) == 1:
                # if self.arm.
                self.this_ret = GRASP_MISS
                self.this_error = "GRASP_MISS"
            else:
                self.this_ret = SUCCESS
                self.this_error = "NONE"
        elif self.this_ret == SAFTY_ISSUE:
            self.this_ret = GRASP_SAFTY
            self.this_error = "GRASP_SAFTY"
        
        self.update()
        print(f'[Primitive INFO] ret: {self.this_ret}, error: {self.this_error}')
        print(f'========== Grasp Done ==========')
        return self.this_ret,self.this_error

    
    @time_it
    def rotate(self,rotate_T=1.5,thresholds=None):
        print(f'========== Rotating ... ==========')
        self.this_id = ROTATE
        self.this_param = [rotate_T,0,0]
        if thresholds is None:
            thresholds = self.rotate_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(open_value=280)
        time.sleep(1)

        ## rotate
        print(f'Rotating ...')
        tag = self.arm.rotate_handle_move_j(T=rotate_T, execute_v=10,if_p=True)
        time.sleep(1)

        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        if self.this_ret == NO_SAFTY_ISSUE:
            if tag != 0:
                self.this_ret = ROTATE_IK_FAIL
                self.this_error = "ROTATE_IK_FAIL"
            elif self.CLIP_detection(rgb_img=Image.fromarray(self.camera.capture_rgb()), text_prompt=['handle with gripper grasping','handle without gripper grasping'],if_p=True) == 1:
                self.this_ret = ROTATE_MISS
                self.this_error = "ROTATE_MISS"
            else:
                self.this_ret = SUCCESS
                self.this_error = "NONE"
        elif self.this_ret == SAFTY_ISSUE:
            self.this_ret = ROTATE_SAFTY
            self.this_error = "ROTATE_SAFTY"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_ret}, error: {self.this_error}')
        print(f'========== Rotate Done ==========')
        return self.this_ret,self.this_error

    @time_it
    def unlock(self,unlock_T=1.5,thresholds=None):
        print(f'========== Unlocking ... ==========')
        self.this_id = UNLOCK
        self.this_param = [unlock_T,0,0]
        if thresholds is None:
            thresholds = self.unlock_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(open_value=280)
        time.sleep(2)

        ## unlock
        print(f'Unlocking ...')
        tag1,tag2 = self.arm.unlock_handle_move_p(T=unlock_T, execute_v=10,if_p=True)
        time.sleep(1)

        ## close gripper
        print(f'Closing Gripper ...')
        self.arm.control_gripper(open_value=150)
        time.sleep(3)
        
        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()

        if self.this_ret == NO_SAFTY_ISSUE:
            if tag1 != 0  or tag2 != 0:
                self.this_ret = UNLOCK_IK_FAIL
                self.this_error = "UNLOCK_IK_FAIL"
            elif self.CLIP_detection(rgb_img=Image.fromarray(self.camera.capture_rgb()), text_prompt=['handle with gripper grasping','handle without gripper grasping'],if_p=True) == 1:
                self.this_ret = UNLOCK_MISS
                self.this_error = "UNLOCK_MISS"
            else:
                self.this_ret = SUCCESS
                self.this_error = "NONE"
        elif self.this_ret == SAFTY_ISSUE:
            self.this_ret = UNLOCK_SAFTY
            self.this_error = "UNLOCK_SAFTY"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_ret}, error: {self.this_error}')
        print(f'========== Unlock Done ==========')
        return self.this_ret,self.this_error

    
    @time_it
    def open(self,open_T=2.0,thresholds=None):
        print(f'========== Opening ... ==========')
        self.this_id = OPEN
        self.this_param = [open_T,0,0]
        if thresholds is None:
            thresholds = self.open_thresholds
        self.start_current_monitor_thread(thresholds=thresholds)

        print(f'Close Gripper ...')
        self.arm.control_gripper(open_value=150)
        time.sleep(1.5)
        print(f'opening ...')
        start_location = self.base.get_location()
        print(f'start_location:{start_location}')
        self.base.move_T(T=open_T)
        time.sleep(abs(open_T)+1)
        end_location = self.base.get_location()
        print(f'end_location:{end_location}')

        ## SAFTY detection
        self.monitor_running = False
        self.current_monitor_thread.join()
        self.vis_current_data()
        
        if self.this_ret == NO_SAFTY_ISSUE:

            if np.sqrt((start_location[0]-end_location[0])**2+(start_location[1]-end_location[1])**2) < 0.2:
                self.this_ret = OPEN_FAIL
                self.this_error = "OPEN_FAIL"
            elif self.CLIP_detection(rgb_img=Image.fromarray(self.camera.capture_rgb()), text_prompt=['handle with gripper grasping','handle without gripper grasping'],if_p=True) == 1:
                self.this_ret = OPEN_MISS
                self.this_error = "OPEN_MISS"
            else:
                self.this_ret = SUCCESS
                self.this_error = "NONE"
        elif self.this_ret == SAFTY_ISSUE:
            self.this_ret = OPEN_SAFTY
            self.this_error = "OPEN_SAFTY"

        self.update()
        print(f'[Primitive INFO] ret: {self.this_ret}, error: {self.this_error}')
        print(f'========== Open Done ==========')
        return self.this_ret,self.this_error

    @time_it
    def finish(self):
        print(f'========== Finishing ... ==========')
        self.arm.control_gripper(open_value=1000)
        time.sleep(1)
        self.base.move_T(-0.5)
        time.sleep(1)
        self.arm.go_home()
        self.base.move_location([self.start_x,self.start_y,self.start_theta])
        # do not update(just for debug)
        self.this_ret = FINISH
        self.this_error = "FINISH"
        print(f'[Primitive INFO] ret: {self.this_ret}, error: {self.this_error}')
        print(f'========== Finish Done... ==========')
        return self.this_ret,self.this_error
    
    def do_primitive(self,_id,_param):
        primitive_type = self.action2num(_id)
        if primitive_type == PREMOVE:
            self.capture()
            ret,error = self.premove(premove_T=_param[0])
        elif primitive_type == GRASP:
            self.capture(if_d=True,vis=True)
            ret,error = self.grasp(grasp_offset=_param[:3])
        elif primitive_type == ROTATE:
            self.capture()
            ret,error = self.rotate(rotate_T=_param[0])
        elif primitive_type == UNLOCK:
            self.capture()
            ret,error = self.unlock(unlock_T=_param[0])
        elif primitive_type == OPEN:
            self.capture()
            ret,error = self.open(open_T=_param[0])
        elif primitive_type == FINISH:
            ret,error = self.finish()
        return ret,error

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