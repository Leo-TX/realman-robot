'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-11 14:36:33
Version: v1
File: 
Brief: 
'''
import paramiko

# Create an SSH client
client = paramiko.SSHClient()

# Automatically add the server's host key (this is insecure and should be avoided in production)
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the server
try:
    client.connect(hostname='130.126.136.95', username='zhi', password='yourpassword')
    print("Connected successfully!")
    
    # Perform operations on the server here
    
    # Disconnect from the server
    client.close()
    print("Disconnected.")
    
except paramiko.AuthenticationException:
    print("Authentication failed. Please check your credentials.")
except paramiko.SSHException as ssh_ex:
    print("Error occurred while connecting or establishing an SSH session:", str(ssh_ex))
except paramiko.ssh_exception.NoValidConnectionsError as conn_ex:
    print("Unable to connect to the server:", str(conn_ex))
except Exception as ex:
    print("An error occurred:", str(ex))