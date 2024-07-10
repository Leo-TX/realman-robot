from PIL import Image
import os

def images2gif(folder_path,save_path=None,duration=200):
    images = []
    # Open all images
    files = os.listdir(folder_path)

    ## sort
    # Sort the file names in ascending order
    # files.sort()

    # Sort the file names in a specific order
    def extract_numeric_part(filename):
        file_number = int(filename.split('.png')[0].split('temp')[1])
        return file_number
    files.sort(key=extract_numeric_part)

    for filename in files:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            images.append(img)
    images[0].save(save_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
    print(f'Create Gif Successfully! Path: {save_path}')

def main():
    # folder_path = r'E:\realman-robot\src\gif-locked'
    # folder_path = r'E:\realman-robot\src\gif-unlocked'
    folder_path = r'E:\realman-robot\src\gif-gelsight-grasping'

    # save_path = 'locked.gif'
    # save_path = 'unlocked.gif'
    save_path = 'gelsight-grasping.gif'

    duration = 200
    images2gif(folder_path,save_path,duration)

if __name__ == "__main__":
    main()