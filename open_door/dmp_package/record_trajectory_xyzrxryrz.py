'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-09 19:22:47
Version: v1
File: 
Brief: 
'''
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import threading
import os

import sys
root_dir = '../'
sys.path.append(root_dir)
from arm import Arm
tjt_path = f'./example_data/data4/refer_tjt.csv'
if not os.path.exists(os.path.dirname(tjt_path)):
    os.makedirs(os.path.dirname(tjt_path))

print ('Program started')
arm= Arm.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_arm_left.yaml')

# ## for right
# initial_pos = [0.008727000094950199, -0.1794009953737259, -0.7527850270271301, -3.069000005722046, 0.050999999046325684, -0.5910000205039978]
# middle_pose = [0.42829400300979614, -0.3133080005645752, -0.127469003200531, -1.5019999742507935, -1.1169999837875366, -2.009000062942505]
# goal_pos = [0.5977209806442261, -0.3816089928150177, -0.13437800109386444, -1.6299999952316284, -1.0529999732971191, -1.9520000219345093]

## for left
initial_pos = [0.7207099795341492, -0.20441000163555145, 0.2483779937028885, -1.5809999704360962, 0.515999972820282, -2.0369999408721924]
middle_pose = [0.08501899987459183, -0.42847099900245667, 0.375230997800827, -1.1480000019073486, 0.5239999890327454, -3.125999927520752]
goal_pos = [0.0728359967470169, -0.5675070285797119, 0.4437209963798523, -1.0709999799728394, 0.4970000088214874, -3.072000026702881]

pos_record_x = list()
pos_record_y = list()
pos_record_z = list()
pos_record_rx = list()
pos_record_ry = list()
pos_record_rz = list()
record_enable = False
data_lock = threading.Lock()  # Create a lock to protect data access

pos_record_x.append(initial_pos[0])
pos_record_y.append(initial_pos[1])
pos_record_z.append(initial_pos[2])
pos_record_rx.append(initial_pos[3])
pos_record_ry.append(initial_pos[4])
pos_record_rz.append(initial_pos[5])

# --- Function to collect data ---
def collect_data():
    global pos_record_x, pos_record_y, pos_record_z, pos_record_rx, pos_record_ry, pos_record_rz, record_enable
    while True:
        # get the currten position
        current_pos = arm.get_p()
        print(f'current_pos: {current_pos}')
        # if (record_enable == False) and (np.sqrt((current_pos[0] - initial_pos[0])**2 + (current_pos[1] - initial_pos[1])**2 + (current_pos[2] - initial_pos[2])**2) < 0.005):
        if (record_enable == False):
            if (np.sqrt((current_pos[0] - initial_pos[0])**2 + (current_pos[1] - initial_pos[1])**2 + (current_pos[2] - initial_pos[2])**2) > 0.005):
                record_enable = True
                print('find a point')
            else:
                print('wait for moving beyond the initial pos')

        if (np.sqrt((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2 + (current_pos[2] - goal_pos[2])**2) < 0.005):
            record_enable = False
            print('reach the goal pos')
            break

        if record_enable == True:
            pos_record_x.append(current_pos[0])
            pos_record_y.append(current_pos[1])
            pos_record_z.append(current_pos[2])
            pos_record_rx.append(current_pos[3])
            pos_record_ry.append(current_pos[4])
            pos_record_rz.append(current_pos[5])
            print('record a point')


# --- Create and start the data collection thread ---
data_thread = threading.Thread(target=collect_data)
data_thread.start()

# --- Initial Movements (Will happen concurrently with data collection) ---
# arm.go_home()
arm.move_p(middle_pose)
arm.move_p(goal_pos)
record_enable = True  # Data recording will start now

# --- Wait for the data collection thread to finish (you'll likely need a different exit condition here) ---
data_thread.join()

pos_record_x.append(goal_pos[0])
pos_record_y.append(goal_pos[1])
pos_record_z.append(goal_pos[2])
pos_record_rx.append(goal_pos[3])
pos_record_ry.append(goal_pos[4])
pos_record_rz.append(goal_pos[5])

print(f'pos number: {len(pos_record_x)}')
data = np.vstack((pos_record_x, pos_record_y, pos_record_z,pos_record_rx,pos_record_ry,pos_record_rz))
# print(data)
df = pd.DataFrame(data)
df.to_csv(tjt_path, index=False, header=None)
print('Program terminated')