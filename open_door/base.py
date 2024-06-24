'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-24 18:04:00
Version: v1
File: 
Brief: 
'''
import socket
import time
import os
import sys
import select
import json

class Base(object):
    def __init__(self,host_ip,host_port,linear_velocity,angular_velocity):
        self.host_ip = host_ip
        self.host_port = host_port
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.connect()

    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('==========\nBase Connecting...')
        self.client_socket.connect((self.host_ip, self.host_port))
        print('Base Connected\n==========')
    
    def move_forward(self,vel=None):
        if not vel:
            vel = self.linear_velocity
        vel = abs(vel)
        command = f"/api/joy_control?angular_velocity=0&linear_velocity={vel}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_back(self,vel=None):
        if not vel:
            vel = self.linear_velocity
        vel = -abs(vel)
        command = f"/api/joy_control?angular_velocity=0&linear_velocity={vel}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_left(self,vel=None):
        if not vel:
            vel = self.angular_velocity
        vel = abs(vel)
        command = f"/api/joy_control?angular_velocity={vel}&linear_velocity=0"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_right(self,vel=None):
        if not vel:
            vel = self.angular_velocity
        vel = -abs(vel)
        command = f"/api/joy_control?angular_velocity={vel}&linear_velocity=0"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)

    def move_stop(self):
        command = f"/api/estop"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)


    def move_char(self,char):
        if char == "w":
            self.move_forward()
        elif char == "s":
            self.move_back()
        elif char == "a":
            self.move_left()
        elif char == "d":
            self.move_right()
        elif char == "x":  # Stop
            self.move_forward(0)
        else:
            pass  # Ignore other keys

    def move_T(self,T,if_p=False):
        start_time = time.time()
        num = 0
        while True:
            if T<=0:
                self.move_char(char='s')
            else:
                self.move_char(char='w')
            num+=1
            time.sleep(0.01)
            if if_p:
                print(f'[Time]: {time.time() - start_time}')
            if time.time() - start_time > abs(T):
                self.move_stop()
                break

    def move_keyboard_win(self, interval=0.1):
        import msvcrt
        def getch():
            char = msvcrt.getch().decode('utf-8')
            return char
        while True:
            try: 
                if msvcrt.kbhit():
                    char = getch()
                    self.move_char(char)
                    time.sleep(interval)  # Adjust delay as needed
            except KeyboardInterrupt:  # Allow Ctrl+C to exit
                break

    def move_keyboard_linux(self, interval=0.1):
        def getch():
            import sys
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                char = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return char
        while True:
            try: 
                char = getch()
                self.move_char(char)
                time.sleep(interval)  # Adjust delay as needed
            except KeyboardInterrupt:  # Allow Ctrl+C to exit
                    break
    
    def move_keyboard(self,interval=0.1):
        if os.name == 'nt':  # Windows
            self.move_keyboard_win(interval)
        else:  # Linux
            self.move_keyboard_linux(interval)

    def move_location(self,location,distance_tolerance=0.5,theta_tolerance=0.05):
        x,y,theta = location
        command = f"/api/move?location={x},{y},{theta}&distance_tolerance={distance_tolerance}&theta_tolerance={theta_tolerance}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def move_marker(self, marker_name,distance_tolerance=0.5,theta_tolerance=0.05):
        command = f"/api/move?marker={marker_name}&distance_tolerance={distance_tolerance}&theta_tolerance={theta_tolerance}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def get_location(self,if_p=False):
        command = f"/api/robot_status"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(9192).decode()
        # print("response:", response)
        robot_status = json.loads(response)
        robot_status = robot_status['results']
        current_floor = robot_status['current_floor']
        current_pose = robot_status['current_pose']
        x = current_pose['x']
        y = current_pose['y']
        theta = current_pose['theta']
        location = [x,y,theta]
        if if_p:
            print(f'location: {location}')
        return location

    def insert_marker(self,marker_name='1311'):
        command = f"/api/markers/insert?name={marker_name}"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(1024).decode()
        # print("response:", response)
    
    def check_marker(self,marker_name='1311'):
        command = f"/api/markers/query_list"
        self.client_socket.send(command.encode('utf-8'))
        response = self.client_socket.recv(9192).decode()
        # print("response:", response)
        results = json.loads(response)
        results = results['results']
        marker = results[marker_name]
        print(f'{marker_name}: {marker}')

    def disconnect(self):
        self.client_socket.close()

if __name__ == "__main__":
    ## init
    host_ip = '192.168.10.10'
    host_port = 31001
    linear_velocity = 0.2
    angular_velocity = 0.2 #　0.2 for slow 1.0 for fast
    base = Base(host_ip,host_port,linear_velocity,angular_velocity)

    ## test move keyboard
    base.move_keyboard(interval=0.1)
    # base.move_T(T=4) # 4 back to starting and -5 for in front of the door

    ## test move T
    # base.move_T(T=5.0)

    # base.get_location()
    # time.sleep(0.5)
    # base.move_char('a')
    # time.sleep(1)

    # base.get_location()
    # location = [-24.1811, 27.3714, 2.6345]
    # base.move_location(location)
    # base.get_location()

    # base.insert_marker(marker_name='1311')
    # base.check_marker(marker_name='1311')
    # base.move_marker(marker_name='1311')

    ## disconnct
    # base.disconnect()