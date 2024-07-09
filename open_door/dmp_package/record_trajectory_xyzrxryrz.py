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
tjt_path = f'./example_data/data3/refer_tjt.csv'
if not os.path.exists(os.path.dirname(tjt_path)):
    os.makedirs(os.path.dirname(tjt_path))

print ('Program started')
arm = Arm('192.168.10.18',8080,root_dir=root_dir)

# ## for right
# initial_pos = [0.008727000094950199, -0.1794009953737259, -0.7527850270271301, -3.069000005722046, 0.050999999046325684, -0.5910000205039978]
# middle_pose = [0.42829400300979614, -0.3133080005645752, -0.127469003200531, -1.5019999742507935, -1.1169999837875366, -2.009000062942505]
# goal_pos = [0.5977209806442261, -0.3816089928150177, -0.13437800109386444, -1.6299999952316284, -1.0529999732971191, -1.9520000219345093] #(second point)

## for left
initial_pos = [0.038137998431921005, 0.24267399311065674, -0.7524719834327698, 3.128999948501587, 0.01899999938905239, -0.3230000138282776]
middle_pose = [0.46181800961494446, 0.39524099230766296, -0.1702989935874939, -1.6469999551773071, -1.0670000314712524, -1.0640000104904175]
goal_pos = [0.5924130082130432, 0.4617370069026947, -0.15729199349880219, -1.5269999504089355, -1.059999942779541, -1.156999945640564] #(second point)

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
arm.move_p(middle_pose, vel=10)
arm.move_p(goal_pos, vel=10)
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