'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-24 18:14:07
Version: v1
File: 
Brief: 
'''
## input like: 
# premove, 1.1
# grasp, -0.04,0.03,0.01
# rotate, 1.8
# unlock, 1.8
# open, 3.0

import argparse
import sys
root_dir = './open_door'
sys.path.append(root_dir)
from primitive import Primitive

def data_collection2(tjt_num):
    primitive = Primitive(root_dir,tjt_num=tjt_num)
    primitive.data_collection2()

def main(args):
    data_collection2(tjt_num=args.tjt_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--tjt_num",type=int,default=6,help="trajectory number.")
    main(parser.parse_args())