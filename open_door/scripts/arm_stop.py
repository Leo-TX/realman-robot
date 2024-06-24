from arm import Arm
arm = Arm('192.168.10.19',8080,cam2base_H_path='cfg/cam2base_H.csv',if_gripper=True,if_monitor=False)# 18 for left 19 for right
arm.move_stop()