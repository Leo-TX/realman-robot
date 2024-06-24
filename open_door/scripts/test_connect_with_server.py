import paramiko

hostname = '130.126.136.95'
username = 'zhi'
password = 'yourpassword'
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
remote_script_path = '/media/datadisk10tb/leo/projects/realman-robot/test/test_server_output.py'


try:
    client.connect(hostname=hostname, username=username, password=password)
    stdin, stdout, stderr = client.exec_command(f'python3 {remote_script_path}') #command
    output = stdout.read().decode('utf-8')
    print(output)
finally:
    client.close()



# # 要在远程服务器上执行的脚本内容
# script = '''
# import os

# # 在此处编写你的脚本逻辑
# # 例如，处理数据的代码
# data = [1, 2, 3, 4, 5]
# result = sum(data)

# # 打印结果
# print("Sum:", result)
# '''

# # 创建SSH客户端
# client = paramiko.SSHClient()
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# try:
#     # 连接远程服务器
#     client.connect(hostname=hostname, username=username, password=password)

#     # 在远程服务器上创建临时脚本文件
#     remote_script_path = '/tmp/script.py'
#     with client.open_sftp() as sftp:
#         with sftp.open(remote_script_path, 'w') as remote_file:
#             remote_file.write(script)

#     # 在远程服务器上执行脚本
#     stdin, stdout, stderr = client.exec_command(f'python {remote_script_path}')

#     # 获取命令执行结果
#     output = stdout.read().decode('utf-8')

#     # 打印命令执行结果
#     print(output)

# finally:
#     # 关闭SSH连接
#     client.close()