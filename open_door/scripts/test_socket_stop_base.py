'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-04-17 18:53:37
Version: v1
File: 
Brief: 
'''
import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.10.10'
port = 31001
print('connecting...')
client.connect((host, port))
print('connected')

command = "/api/estop?flag=false"
client.send(command.encode('utf-8'))
response = client.recv(1024).decode()
print(response)

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