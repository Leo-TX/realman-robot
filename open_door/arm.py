'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-03 09:01:25
Version: v1
File: 
Brief: 
'''
#coding=utf8
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import socket
import threading

from utils.math import *
from arm_package.robotic_arm import Arm as ArmBase

VZ_SPEED = 0.02 # 2cm/s
VZ_SPEED_DEGREE = 14 # 14°/s
VYAW_SPEED_DEGREE = 35 # 35°/s
VYAW_SPEED_RADIAN = 25*np.pi/180

class ArmInfoMonitor:
    def __init__(self, arm, refresh_freq=1, title='', gif_folder='./images/image1/current_monitor'):
        self.arm = arm
        self.title = title
        self.refresh_freq = refresh_freq
        self.gif_folder = gif_folder
        self.fig, self.ax = plt.subplots()
        self.times = []
        self.values = [[] for _ in range(7)]  # List of empty lists for each joint
        self.lines = []  # List to store line objects
        self.plot_started = False
        self.ax.legend()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel(self.title+' Value')
        self.ax.set_title('Arm Information - ' + title)
        self.images = []
        self.num = 0
        if not os.path.exists(self.gif_folder):
            os.mkdir(self.gif_folder)

    def update_plot(self):
        print(f'===update')
        self.ax.clear()
        for joint in range(7):
            self.ax.plot(self.times, self.values[joint], label=f'Joint {joint+1}')
        self.ax.legend(loc='upper left')  # Set the legend location to upper left
        self.ax.set_xlabel('Time/s')
        self.ax.set_ylabel(self.title+' Value')
        self.ax.set_title(self.ax.get_title())
        self.fig.canvas.draw()

        # Save the current plot as an image
        self.fig.savefig(self.gif_folder+f'/info_img_{self.num}.png', format='png')
        # Open the saved image
        img = Image.open(self.gif_folder+f'/info_img_{self.num}.png')
        # Append the image to a list
        self.images.append(img)
        # # close image
        # img.close()
        # # Remove the temporary image file
        # os.remove(f'gif/info_img_{self.num}.png')
        self.num += 1

    def detect_info_realtime(self):
        while True:
            if self.title == 'current':
                tag, values = self.arm.Get_Joint_Current()
            elif self.title == 'voltage':
                tag, values = self.arm.Get_Joint_Voltage()
            elif self.title == 'joint':
                tag, values = self.arm.Get_Joint_Degree()
            else:
                print(f'Wrong Parameter!')
            # print(f'=== info: {values}')
            for joint in range(7):
                self.values[joint].append(values[joint])

            self.times.append(time.time())

            if not self.plot_started:
                self.plot_started = True
                plt.ion()  # Turn on interactive mode for dynamic updating
                # plt.show()

            if len(self.times) > 1:  # Update plot only after the first data point
                self.update_plot()

            plt.pause(1 / self.refresh_freq)

            # Save the list of images as a GIF
            if os.path.exists(self.gif_folder+'/current_monitor.gif'):
                os.remove(self.gif_folder+'/current_monitor.gif')
            if len(self.images) > 2:
                self.images[0].save(self.gif_folder+'/current_monitor.gif', save_all=True, append_images=self.images[1:], optimize=False, duration=200, loop=0)

class Arm():
    def __init__(self, host_ip, host_port, cam2base_H_path=None, home_state=[0,-90,0,0,0,0,0], workspace_limits=[[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]],if_gripper=False,if_monitor=False,gif_folder=None,tool_frame='dh3'):
        self.host_ip = host_ip
        self.host_port = host_port
        self.home_state = home_state
        self.home_state1 = [-4.303,-95.307,-80.395,12.244,3.457,-8.106,20.592] #pose: [0.008727000094950199, -0.1794009953737259, -0.7527850270271301, -3.069000005722046, 0.050999999046325684, -0.5910000205039978]
        self.workspace_limits = workspace_limits
        self.if_gripper = if_gripper
        self.tool_frame = tool_frame
        if cam2base_H_path:
            with open(cam2base_H_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                data = []
                for row in reader:
                    data.append(row)
            self.cam2base_H = np.array(data, dtype=np.float32)
        else:
            self.cam2base_H = None
        # connect
        self.connect()
        if if_gripper:
            self.connect_gripper()
        if if_monitor:
            self.current_monitor = ArmInfoMonitor(self.arm,refresh_freq=5, title='current',gif_folder=gif_folder)
            # self.voltage_monitor = ArmInfoMonitor(self.arm,refresh_freq=5, title='voltage')
            # self.joint_monitor = ArmInfoMonitor(self.arm,refresh_freq=5, title='joint')
        
    def __str__(self):
        self.get_j()
        self.get_p()
        self.get_v()
        self.get_c()
        self.get_api_version()
        return ''

    def connect(self):
        print('==========\nArm Connecting...')
        self.arm = ArmBase(self.host_ip,self.host_port)
        self.change_tool_frame(self.tool_frame)
        print('Arm Connected\n==========')


    def disconnect(self):
        self.arm.Arm_Socket_Close()
    
    def get_api_version(self):
        self.arm.API_Version()

    def start_current_monitor(self):
        self.current_monitor.detect_info_realtime()
    

    def connect_gripper(self,force=30,start_pos=1000,vel=30,tool_voltage=3,port=1,baudrate=115200,device=1):
        ADDRESS_INIT_GRIPPER = int(0x0100)
        ADDRESS_SET_FORCE = int(0x0101)
        ADDRESS_SET_POS = int(0X0103)
        ADDRESS_SET_VEL = int(0X0104)
        ADDRESS_GET_GRIPPER_INIT_RETURN = int(0x0200)
        ADDRESS_GET_GRIPPER_GRASP_RETURN = int(0x0201)
        ADDRESS_GET_GRIPPER_POS = int(0x0202)

        print(f'ADDRESS_INIT_GRIPPER:{ADDRESS_INIT_GRIPPER}')
        tag = self.arm.Set_Tool_Voltage(type=tool_voltage,block=True)
        tag = self.arm.Set_Modbus_Mode(port=port, baudrate=baudrate, timeout=2, block=True)
        tag = self.arm.Write_Single_Register(port=port, address=ADDRESS_INIT_GRIPPER, data=1, device=device, block=True)
        tag = self.arm.Write_Single_Register(port=port, address=ADDRESS_SET_FORCE, data=force, device=device, block=True)
        tag = self.arm.Write_Single_Register(port=port, address=ADDRESS_SET_POS, data=start_pos, device=device, block=True)
        tag = self.arm.Write_Single_Register(port=port, address=ADDRESS_SET_VEL, data=vel, device=device, block=True)
        tag, value = self.arm.Get_Read_Input_Registers(port=port, address=ADDRESS_GET_GRIPPER_INIT_RETURN, device=device)
        if tag != 1: #0: not init. 1: init is successful. 2: initializing
            print(f'init failed: {value}\n')
        return tag

    def control_gripper(self,open_value,port=1,device=1):
        ADDRESS_SET_POS = int(0X0103)
        tag = self.arm.Write_Single_Register(port=port, address=ADDRESS_SET_POS, data=open_value, device=device, block=True)
        return tag
    def get_gripper_grasp_return(self,port=1,device=1):
        ADDRESS_GET_GRIPPER_GRASP_RETURN = int(0x0201)
        tag, value = self.arm.Get_Read_Input_Registers(port=port, address=ADDRESS_GET_GRIPPER_GRASP_RETURN, device=device)
        return value # 0 for moving; 1 for detecting no objects grasping; 2 for detecting objects grasping; 3 for detecting objecting dropped after detecting grasping

    def get_gripper_pos(self,port=1,device=1):
        ADDRESS_GET_GRIPPER_POS = int(0x0202)
        tag, value = self.arm.Get_Read_Input_Registers(port=port, address=ADDRESS_GET_GRIPPER_POS, device=device)
        return value # 0-1000

    def go_home(self):
        self.move_j(joint=self.home_state1,vel=20)
    
    def get_p(self,if_p=False):
        pose = self.arm.Get_Current_Pose()
        if if_p:
            print(f'[Arm INFO]: - {self.get_p.__name__}: {pose}')
        return pose

    def get_j(self,if_p=False):
        tag,joint = self.arm.Get_Joint_Degree()
        if if_p:
            print(f'[Arm INFO]: - {self.get_j.__name__}: {joint}')
        return joint

    def get_v(self,if_p=False):
        tag, voltage = self.arm.Get_Joint_Voltage()
        if if_p:
            print(f'[Arm INFO]: - {self.get_v.__name__}: {voltage}')
        return voltage

    def get_c(self,if_p=False):
        tag, current = self.arm.Get_Joint_Current()
        if if_p:
            print(f'[Arm INFO]: - {self.get_c.__name__}: {current}')
        return current
    
    def move_j(self,joint,vel,trajectory_connect=0, r=0, block=True, if_p=False):
        tag = self.arm.Movej_Cmd(joint, vel, trajectory_connect, r, block) 
        if if_p:
            print(f'[Arm INFO]: - {self.move_j.__name__}: {tag}')
        return tag

    def move_p(self,pos,vel,trajectory_connect=0, r=0, block=True, if_p=False):
        tag = self.arm.Movej_P_Cmd(pos, vel, trajectory_connect, r, block)
        if if_p:
            print(f'[Arm INFO]: - {self.move_p.__name__}: {tag}')
        return tag

    def move_l(self,pos,vel,trajectory_connect=0, r=0, block=True, if_p=False):
        tag = self.arm.Movel_Cmd(pos, vel, trajectory_connect, r, block)
        if if_p:
            print(f'[Arm INFO]: - {self.move_l.__name__}: {tag}')
        return tag

    def move_j_with_input(self,frame_name='Arm_Tip'):
        if frame_name != 'Arm_Tip':
            self.arm.Change_Tool_Frame(frame_name)
        while True:
            pose_input = input("Enter the pose (joint angles): ")
            if pose_input == 'q':
                break

            pose_list = [float(num) for num in pose_input.split(',')]

            if len(pose_list) == 7:
                pose_list.append(10)  # Default velocity

            joint_angles = pose_list[:7]
            velocity = int(pose_list[7])
            print(f'joints: {joint_angles}')
            print(f'velocity: {velocity}')
            self.move_j(joint_angles, velocity)

    def move_p_with_input(self,frame_name='Arm_Tip'):
        self.arm.Change_Tool_Frame(frame_name)
        while True:
            pose_input = input("Enter the pose: ")
            if pose_input == 'q':
                break

            pose_list = [float(num) for num in pose_input.split(',')]

            if len(pose_list) == 6:
                pose_list.append(10)  # Default velocity

            pose = pose_list[:6]
            # for i in range(3):
            #     pose[i] = pose[i] / 1000 # mm to m

            velocity = int(pose_list[6])
            print(f'pose: {pose}')
            print(f'velocity: {velocity}')
            self.move_p(pose, velocity)

    def move_l_with_input(self,frame_name='Arm_Tip'):
        self.arm.Change_Tool_Frame(frame_name)
        while True:
            pose_input = input("Enter the pose: ")
            if pose_input == 'q':
                break

            pose_list = [float(num) for num in pose_input.split(',')]

            if len(pose_list) == 6:
                pose_list.append(10)  # Default velocity

            pose = pose_list[:6]
            # for i in range(3):
            #     pose[i] = pose[i] / 1000 # mm to m

            velocity = int(pose_list[6])
            print(f'pose: {pose}')
            print(f'velocity: {velocity}')
            self.move_l(pose, velocity)

    def move_poses(self,poses,vel=10,trajectory_connect=1,frame_name='Arm_Tip',if_p=False):
        self.arm.Change_Tool_Frame(frame_name)
        for pos in poses:
            self.move_p(pos=pos, vel=vel,trajectory_connect=trajectory_connect, r=0, block=True,if_p=if_p)

    def move_stop(self, if_p=False):
        tag = self.arm.Move_Stop_Cmd(block=True)
        if if_p:
            print(f'[Arm INFO]: - {self.move_stop.__name__}: {tag}')

    def else1(self,joint_diff=[0,5,0,0,0,-20,-45],vel=3): #overunlock:[0,8,0,0,0,-25,-50]
        joint = self.get_j(if_p=False)
        for i in range(7):
            joint[i] += joint_diff[i]
        self.arm.move_j(joint,vel)

    def rotate_handle_move_teach(self, T=1.0, v=30,if_p=False):
        start_time = time.time()
        while True:
            # self.arm.Ort_Teach_Cmd(type, direction, v, block)
            self.arm.Joint_Teach_Cmd(num=7, direction=0, v=100, block=0) # 20 will convulsions
            time.sleep(0.05)
            print(f'[Time]: {time.time() - start_time}')
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > T:
                self.arm.Teach_Stop_Cmd()
                print(f'Arm Stop!!!')
                break
    
    def rotate_handle_move_j(self, T=1.0, execute_v=20,if_p=False):
        joint_goal = self.get_j()
        joint_goal[6] = joint_goal[6] - T*VYAW_SPEED_DEGREE
        tag = self.move_j(joint=joint_goal,vel=execute_v,if_p=if_p)
        return tag

    def rotate_handle_move_p(self, T=1.0, execute_v=20,if_p=False):
        pos_goal = self.get_p()
        pos_goal[2] = pos_goal[2] - T*VYAW_SPEED_RADIAN
        tag = self.move_p(pos=pos_goal,vel=execute_v,if_p=if_p)
        return tag

    def unlock_handle_move_teach(self, T=1.0,v=30,if_p=False):
        start_time = time.time()
        self.arm.Start_Force_Position_Move()
        num = 0
        while True:
            self.arm.Joint_Teach_Cmd(num=7, direction=0, v=v, block=0)
            self.arm.Pos_Teach_Cmd(type=2, direction=0, v=v, block=0)
            num+=1
            time.sleep(0.5)
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > T:
                self.arm.Teach_Stop_Cmd()
                print(f'Arm Stop!!!')
                print(f'Num: {num}')
                break

    def unlock_handle_move_j(self, T=1.0, execute_v=20,if_p=False):
        joint_goal = self.get_j()
        joint_goal[5] = joint_goal[5] - T*VZ_SPEED_DEGREE
        joint_goal[6] = joint_goal[6] - T*VYAW_SPEED_DEGREE
        tag = self.move_j(joint=joint_goal,vel=execute_v,if_p=if_p)
        return tag

    def unlock_handle_move_p(self, T=1.0, execute_v=20,if_p=False):
        num=1
        z_diff = T*VZ_SPEED
        yaw_diff = T*VYAW_SPEED_DEGREE
        for i in range(num):
            joint = self.get_j()
            joint[6] -= yaw_diff*(i+1)/num
            tag1 = self.move_j(joint=joint,vel=execute_v,if_p=if_p)
            pos = self.get_p()
            pos[2] -= z_diff*(i+1)/num
            tag2 = self.move_p(pos=pos,vel=execute_v,if_p=if_p)
        return tag1,tag2

    def target2cam_xyzrpy_to_target2base_xyzrpy(self,target2cam_xyzrpy,if_gripper=True):
        cam2base_H = self.cam2base_H
        # print(f'cam2base_H:\n{cam2base_H}')

        target2cam_R = EulerAngle_to_R(np.array(target2cam_xyzrpy[3:]),rad=True)
        target2cam_t = xyz_to_t(np.array(target2cam_xyzrpy[:3]))
        target2cam_H = Rt_to_H(target2cam_R, target2cam_t)
        # print(f'target2cam_H:\n{target2cam_H}')

        target2base_H = cam2base_H @ target2cam_H
        # print(f'target2base_H:\n{target2base_H}')

        if if_gripper:
            tcp2base_H = self.gripper2base_H_to_tcp2base_H(gripper2base_H=target2base_H)
            target2base_H = tcp2base_H

        target2base_xyzrpy = H_to_xyzrpy(target2base_H,rad=True)
        # print(f'target2base_xyzrpy:\n{target2base_xyzrpy}')

        return target2base_xyzrpy.tolist()

    def move_to_target(self,target2cam_xyzrpy):
        target2base_xyzrpy = self.point_cam2base_xyzrpy(target2cam_xyzrpy)
        self.move_p(target2base_xyzrpy)

    def gripper2base_H_to_tcp2base_H(self,gripper2base_H):
        tcp2gripper_xyzrpy = [0,0,-0.180,0,0,0]
        tcp2gripper_H = xyzrpy_to_H(tcp2gripper_xyzrpy,rad=True)
        tcp2base_H = tcp2gripper_H @ gripper2base_H
        return tcp2base_H

    def tcp2base_H_to_gripper2base_H(self,tcp2base_H):
        tcp2gripper_xyzrpy = [0,0,-0.180,0,0,0]
        tcp2gripper_H = xyzrpy_to_H(tcp2gripper_xyzrpy,rad=True)
        gripper2base_H = tcp2base_H @ np.linalg.inv(tcp2gripper_H)
        return gripper2base_H

    def get_current_tool_frame(self,if_p=False):
        tag, frame = self.arm.Get_Current_Tool_Frame()
        if if_p:
            print(f'current tool frame:')
            self.arm.print_frame(frame)
        return frame

    def get_all_tool_frame(self,if_p=False):
        tag, tool_names, tool_len = self.arm.Get_All_Tool_Frame()
        if if_p:
            print(f'all tool names:{tool_names}')
        return tool_names

    def get_given_tool_frame(self,tool_name,if_p=False):
        tag, frame = self.arm.Get_Given_Tool_Frame(tool_name)
        if if_p:
            print(f'given tool frame:')
            self.arm.print_frame(frame)
        return frame

    def change_tool_frame(self,tool_name,if_p=False):
        self.arm.Change_Tool_Frame(tool_name)
        tag, frame = self.arm.Get_Current_Tool_Frame()
        if if_p:
            print(f'current tool frame:')
            self.arm.print_frame(frame)
        return frame

    def manual_set_tool_frame(self,tool_name,pose=[0,0,0.16,0,0,0],payload=0, x=0, y=0, z=0, block=True,if_p=True):
        self.arm.Manual_Set_Tool_Frame(tool_name, pose, payload, x, y, z, block)
        frame = self.get_given_tool_frame(self,tool_name,if_p=False)
        if if_p:
            print(f'set a new tool frame:')
            self.arm.print_frame(frame)
        return frame


    def else_functions(self):
        # arm.Save_Trajectory(file_name='./test_tjt.txt')
        # arm.MoveCartesianTool_Cmd(joint_cur=joint,movelengthx=0.000, movelengthy=0.00, movelengthz=0.00, m_dev=75, v=3, trajectory_connect=0, r=0)
        # arm.MoveRotate_Cmd(rotateAxis=3, rotateAngle=90, choose_axis=[0,0,0,0,0,0], v=3, trajectory_connect=0, r=0, block=True)
        # arm.Movec_Cmd(self, pose_via, pose_to, v, loop, trajectory_connect=0, r=0, block=True)
        pass

if __name__ =="__main__":
    ## connect
    arm = Arm('192.168.10.19',8080,cam2base_H_path='cfg/cam2base_H.csv',if_gripper=True,if_monitor=False,tool_frame='dh3')# 18 for left 19 for right
    print(arm)
    arm.go_home()
    # arm.get_c(if_p=True)
    # arm.control_gripper(open_value=1000)
    # arm.get_p(if_p=True)
    # arm.move_p(pos=[0.6051296976350004, -0.35271136656822955, -0.1534878647732853, -1.536159878101188, -1.0257231725488505, -1.9371845128439396],vel=10)
    # arm.unlock_handle_move_j(T=1.8, execute_v=5)
    # arm.arm.Joint_Teach_Cmd(num=7, direction=0, v=30, block=0)
    # arm.arm.Pos_Teach_Cmd(type=2, direction=0, v=30, block=0)
    # arm.unlock_handle_move_teach(T=1.0,v=60,if_p=False)
    # arm.unlock_handle_move_p( T=1.8, execute_v=5)


    ## info: current monitor
    # arm = Arm('192.168.10.19',8080,if_gripper=False,if_monitor=True,gif_folder='./images/image1/current_monitor/')
    # arm.start_current_monitor()

    ## disconnect
    arm.disconnect()