import os
import sys
import glob
import yaml
import csv
import simplejson
import time
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

def get_filenames(folder, is_base_name=False, filter=None): # filter: 'png' ,'txt' ...
    ''' Get all filenames under the specific folder. 
    e.g.:
        full name: data/rgb/000001.png
        base name: 000001.png 
    '''
    full_names = sorted(glob.glob(folder + "/*"))
    if is_base_name:
        base_names = [name.split("/")[-1] for name in full_names]
        if filter:
            base_names = [name for name in base_names if name.endswith(filter)]
        return base_names
    else:
        if filter:
            full_names = [name for name in full_names if name.endswith(filter)]
        return full_names

def rename_files_sequentially(folder):
    """Renames all files in a folder sequentially starting from 0.

    Args:
        folder (str): The path to the folder containing the files.
    """

    files = sorted(os.listdir(folder))
    for i, file in enumerate(files):
        old_path = os.path.join(folder, file)
        extension = os.path.splitext(file)[1]
        new_file = f"{i}{extension}"
        new_path = os.path.join(folder, new_file)
        os.rename(old_path, new_path)
        # print(f"Renamed '{file}' to '{new_file}'")

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, (dict)):
                setattr(self, key, self.__class__(value))
            else:
                setattr(self, key, value)

def getch_win():
    import msvcrt
    char = msvcrt.getch()
    # special char
    if char == b'\xe0':
        return {
            b'U': "up",
            b'P': "down",
            b'K': "left",
            b'M': "right",
        }.get(char, None)
    # normal char
    else:
        return char.decode('utf-8')

def getch_linux():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char
    
def getch(if_p=False):
    if os.name == 'nt':  # Windows
       char = getch_win()
    else:  # Linux
       char = getch_linux()
    if if_p:
        print(f'char: {char}')
    return char

if __name__ == "__main__":
    interval = 0.1
    while True:
        try: 
            char = getch(if_p=True)
            time.sleep(interval)  # Adjust delay as needed
            if char == 'q':
                break
            if char == '0':
                print('000')
        except KeyboardInterrupt:  # Allow Ctrl+C to exit
            break