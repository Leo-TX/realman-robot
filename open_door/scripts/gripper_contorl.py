'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-18 16:48:53
Version: v1
File: 
Brief: 
'''
import socket
import time

def send_cmd(client, cmd_6axis):
  client.send(cmd_6axis.encode('utf-8'))
  # Optional: Receive a response from the server
  # _ = client.recv(1024).decode()
  return True

# IP and port configuration
ip = '192.168.10.19' # 18forleft 19for right
port_no = 8080

# Create a socket and connect to the server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, port_no))
print("机械臂第一次连接", ip)
point6_00 = '{"command":"set_tool_voltage","voltage_type":3}\r\n'
_ = send_cmd(client, cmd_6axis=point6_00)
print("//设置工具端电源输出 24V")
time.sleep(1)
point6_00 = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout ":2}\r\n'
_ = send_cmd(client, cmd_6axis=point6_00)
print("配置通讯端口 ModbusRTU 模式")
time.sleep(1)
point6_00 = '{"command":"write_single_register","port":1,"address":256,"data":1, "device":1}\r\n'
_ = send_cmd(client, cmd_6axis=point6_00)
print("  执行初始化成功")
time.sleep(1)
point6_00 = '{"command":"write_single_register","port":1,"address":257,"data":30, "device":1}\r\n'
_ = send_cmd(client, cmd_6axis=point6_00)
print(" 设置30% 力值 （写操作）")
time.sleep(1)

# Loop to wait for user input and control the gripper
while True:
    user_input = input("Enter 'open' to open the gripper or 'close' to close it: ")
    user_input = user_input.lower()  # Convert to lowercase for case-insensitive input

    point6_00 = f'{{"command":"write_single_register","port":1,"address":259,"data":{user_input}, "device":1}}\r\n'
    _ = send_cmd(client, cmd_6axis=point6_00)
    print("success")

    # if user_input == "open":
    #     point6_00 = '{"command":"write_single_register","port":1,"address":259,"data":1000, "device":1}\r\n'
    #     _ = send_cmd(client, cmd_6axis=point6_00)
    #     print("Gripper opened")

    # elif user_input == "close":
    #     point6_00 = '{"command":"write_single_register","port":1,"address":259,"data":100, "device":1}\r\n'
    #     _ = send_cmd(client, cmd_6axis=point6_00)
    #     print("Gripper closed")

    # else:
    #     print("Invalid input. Please enter 'open' or 'close'.")

    # Optional: Add a delay between input prompts (if desired)
    # time.sleep(1)