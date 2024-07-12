import os
import sys
import glob
import yaml
import csv
import simplejson
import numpy as np

def makedirs(output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

def read_yaml_file(file_path, is_convert_dict_to_class=True):
    with open(file_path, 'r') as stream:
        data = yaml.safe_load(stream)
    if is_convert_dict_to_class:
        data = Config(data)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = simplejson.load(f)
    return data

def read_csv_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append(row)
    data_np = np.array(data, dtype=np.float32)
    return data_np

def get_filenames(folder, is_base_name=False):
    ''' Get all filenames under the specific folder. 
    e.g.:
        full name: data/rgb/000001.png
        base name: 000001.png 
    '''
    full_names = sorted(glob.glob(folder + "/*"))
    if is_base_name:
        base_names = [name.split("/")[-1] for name in full_names]
        return base_names
    else:
        return full_names
class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, (dict)):
                setattr(self, key, self.__class__(value))
            else:
                setattr(self, key, value)