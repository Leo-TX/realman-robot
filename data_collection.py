'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-24 18:14:07
Version: v1
File: 
Brief: 
'''
# input like: 
# premove, 1.1
# grasp, -0.04,0.03,0
# rotate, 1.8
# unlock, 1.8
# open, 3.0

import sys
root_dir = './open_door'
sys.path.append(root_dir)
from primitive import Primitive

def data_collection():
    primitive = Primitive(root_dir,tjt_num=5)
    num = 0
    while True:
        num += 1
        print('***************************************************************')
        user_input = input(f"[Please Input Primitive_{num}]: ")
        if user_input.lower() == 'q':
            break
        try:
            action_id, *param = [x.strip() for x in user_input.split(',')]
            param = [float(x) for x in param]
            ret, error = primitive.do_primitive(action_id, param)
        except Exception as e:
            print(f"ERROR TYPE: {type(e)}: {e}")
            print("Please re-input!")
        print('***************************************************************\n\n')

if __name__ =="__main__":
    data_collection()