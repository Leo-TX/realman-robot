'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-18 15:33:11
Version: v1
File: 
Brief: 
'''
'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-17 18:53:37
Version: v1
File: 
Brief: 
'''
import socket
import time
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.10.10'
port = 31001
print('connecting...')
client.connect((host, port))
print('connected')

command = '/api/robot_info'
client.send(command.encode('utf-8'))
response = client.recv(1024).decode()
print(response)

# command = '/api/robot_status'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)

command = '/api/joy_control?angular_velocity=0&linear_velocity=-0.2' 
# angular_velocity:-1 to 1 rad/s
# linear_velocity: -0.5 to 0.5 m/s
for i in range(20):
    client.send(command.encode('utf-8'))        
    time.sleep(0.3)
response = client.recv(1024).decode()
print(response)


# command = "/api/estop?flag=true"
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)

# ======

# command = '/api/markers/insert?name=end_point'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)

# command = '/api/markers/query_list'
# client.send(command.encode('utf-8'))
# # response = client.recv(1024).decode()
# response = client.recv(4096).decode('utf-8')
# print(response)

# command = '/api/markers/delete'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)

# command = '/api/markers/count'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode('utf-8')
# print(response)

# command = '/api/markers/query_brief'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode('utf-8')
# print(response)

# ======

# command= '/api/move?marker=start_point'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)


# command= '/api/move?markers=start_point,end_point'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)

# command= '/api/move/cancel'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)



# command = '/api/map/list'
# client.send(command.encode('utf-8'))
# response = client.recv(1024).decode()
# print(response)