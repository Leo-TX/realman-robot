import argparse
import sys
root_dir = './open_door'
sys.path.append(root_dir)
from primitive import Primitive

def state_machine(tjt_num):
    primitive = Primitive(root_dir,tjt_num=tjt_num)
    primitive.data_collection()

def main(args):
    data_collection(tjt_num=args.tjt_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--tjt_num",type=int,default=6,help="trajectory number.")
    main(parser.parse_args())