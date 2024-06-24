import os
import cv2
import numpy as np
import logging

from save_poses import poses_main
from save_poses2 import poses2_main


def computer_data(now_forder):
    """
    根据采集到的图片和位姿计算相机和机械臂之间的空间关系
    :return:
    """


    #如果图片数量小于15张，提醒增加图片数量后再计算

    matching_pictures = sorted([folder for folder in os.listdir(now_forder) if folder.endswith('.jpg') and folder.split(".")[0].isdigit()],key=lambda x:int(x.split('.')[0]))
    print(matching_pictures)
    if len(matching_pictures) >= 15:

        length,width = (10,7)
        cell_width = 2.18
        pattern = '眼在手外'

        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        # 获取标定板角点的位置
        objp = np.zeros((length * width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:length, 0:width].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        objp = cell_width / 100 * objp

        obj_points = []  # 存储3D点
        img_points = []  # 存储2D点


        for picture in matching_pictures:
            image = f"{now_forder}/{picture}"
            if os.path.exists(image):

                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, (length, width), None)

                if ret:

                    obj_points.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                                criteria)  # 在原角点的基础上寻找亚像素角点
                    if [corners2]:
                        img_points.append(corners2)
                    else:
                        img_points.append(corners)

        N = len(img_points)

        # 标定,得到图案在相机坐标系下的位姿
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

        print(f"内参矩阵:{mtx}")  # 内参数矩阵
        print(f"畸变系数:{dist}" )  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

        if pattern == '眼在手上':

            poses_main(now_forder,f'{now_forder}/poses.txt')

        else:

            poses2_main(now_forder,f'{now_forder}/poses.txt')

        #机器人末端在基座标系下的位姿

        tool_pose = np.loadtxt(f'{now_forder}/RobotToolPose.csv', delimiter=',')
        R_tool = []
        t_tool = []
        for i in range(int(N)):
            R_tool.append(tool_pose[0:3, 4 * i:4 * i + 3])
            t_tool.append(tool_pose[0:3, 4 * i + 3])

        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_TSAI)
        #print(R)
        # print(t)
        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
        print(R)
        print(t)
        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
        #print(R)
        # print(t)
        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
        #print(R)
        # print(t)
        R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_ANDREFF)
        print(R)
        print(t)

    else:
        print("当前采集图片不足15张，请补充采集图片")

def main():
    now_forder = r'E:\realman-robot-2\open_door\arm_package\handeye_calibration\data\2024061301\\'
    computer_data(now_forder)

if __name__ == "__main__":
    main()