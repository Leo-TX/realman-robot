import os
import json
import re
from PIL import Image
import random
import numpy as np

from utils.lib_rgbd import *

class DataAugmentator:
    def __init__(self, input_dir, output_dir, translation_x_min=-30, translation_x_max=30,
                 translation_y_min=-30, translation_y_max=30, translation_num=5,
                 ratio_min=0.8, ratio_max=1.2, resize_num=5):
        """
        Initializes the DataAugmentator class with augmentation parameters.

        Args:
            input_dir (str): The directory containing the input images and JSON files.
            output_dir (str): The directory to save the augmented data.
            translation_x_min (int, optional): Minimum translation in x-axis. Defaults to -50.
            translation_x_max (int, optional): Maximum translation in x-axis. Defaults to 50.
            translation_y_min (int, optional): Minimum translation in y-axis. Defaults to -50.
            translation_y_max (int, optional): Maximum translation in y-axis. Defaults to 50.
            translation_num (int, optional): Number of random translations to perform. Defaults to 5.
            ratio_min (float, optional): Minimum resize ratio. Defaults to 0.8.
            ratio_max (float, optional): Maximum resize ratio. Defaults to 1.2.
            resize_num (int, optional): Number of random resizes to perform. Defaults to 5.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.translation_x_min = translation_x_min
        self.translation_x_max = translation_x_max
        self.translation_y_min = translation_y_min
        self.translation_y_max = translation_y_max
        self.translation_num = translation_num
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.resize_num = resize_num
        self.crop_width = 640
        self.crop_height = 480
        self.num = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def augment_data(self):
        """
        Augments the images and annotations in the input directory.
        """
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".png") and re.match(r'^\d+\.png$', filename):
                image_path = os.path.join(self.input_dir, filename)
                mask_path = os.path.join(self.input_dir, os.path.splitext(filename)[0] + "_mask.png")
                json_path = os.path.join(self.input_dir, os.path.splitext(filename)[0] + ".json")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if 'dx' not in data: # not annotated
                        continue
                    print(f'[num_{self.num}]: processing the {filename}')
                    self.augment_single_image(image_path,mask_path,data)
                    self.num += 1

    def visualization(self,data,new_filename):
        if data['R'] != 0:
            x1_2d, y1_2d = data['Cx'] + data['dx'], data['Cy'] + data['dy']
            dx = data['dx']
            dy = data['dy']
            R = data['R']
            orientation = data['orientation']
            angle = 90
            img_path = os.path.join(self.output_dir, new_filename)
            save_path = img_path.replace('.png', '_vis.png')
            x2_2d, y2_2d, Ox, Oy = rotate_point(x1_2d, y1_2d, R, orientation, angle)
            vis_grasp(img_path, dx, dy, x1_2d, y1_2d, x2_2d, y2_2d, Ox, Oy, R, orientation, angle, save_path)

    def augment_single_image(self, image_path,mask_path,data):
        """
        Augments a single image and its annotations.

        Args:
            image_path (str): Path to the image file.
            data (dict): Dictionary containing the image annotations.
        """
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        original_width, original_height = image.size

        for i in range(self.translation_num):
            for j in range(self.resize_num):
                # Random translation
                tx = random.randint(self.translation_x_min, self.translation_x_max)
                ty = random.randint(self.translation_y_min, self.translation_y_max)

                # Random resize
                ratio = random.uniform(self.ratio_min, self.ratio_max)
                
                # 1. Resize image and annotations
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                resized_image = image.resize((new_width, new_height))
                resized_mask = mask.resize((new_width, new_height))
                resized_data = self.adjust_annotations(data.copy(), 0, 0, ratio)

                # # Save resized image and JSON
                # new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_resized.png"
                # resized_image.save(os.path.join(self.output_dir, new_filename))
                # new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_resized.json"
                # with open(os.path.join(self.output_dir, new_json_filename), 'w') as f:
                #     json.dump(resized_data, f, indent=4)
                # self.visualization(resized_data,new_filename)

                # 2. Translate image and annotations
                translated_data = self.adjust_annotations(resized_data.copy(), tx, ty, 1)
                translated_image = Image.new("RGB", (new_width, new_height))
                translated_mask = Image.new("RGB", (new_width, new_height))
                translated_image.paste(resized_image, (tx, ty))
                translated_mask.paste(translated_mask, (tx, ty))

                # # Save translated image and JSON
                # new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_translated.png"
                # translated_image.save(os.path.join(self.output_dir, new_filename))
                # new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_translated.json"
                # with open(os.path.join(self.output_dir, new_json_filename), 'w') as f:
                #     json.dump(translated_data, f, indent=4)
                # self.visualization(translated_data,new_filename)

                # 3. Calculate crop coordinates
                crop_x_min = translated_data['Cx'] - self.crop_width // 2
                crop_y_min = translated_data['Cy'] - self.crop_height // 2
                crop_x_max = crop_x_min + self.crop_width
                crop_y_max = crop_y_min + self.crop_height

                # Adjust crop coordinates to stay within image boundaries
                if crop_x_min < 0:
                    crop_x_min = 0
                    crop_x_max = self.crop_width
                elif crop_x_max > new_width:
                    crop_x_max = new_width
                    crop_x_min = new_width - self.crop_width

                if crop_y_min < 0:
                    crop_y_min = 0
                    crop_y_max = self.crop_height
                elif crop_y_max > new_height:
                    crop_y_max = new_height
                    crop_y_min = new_height - self.crop_height

                # Crop image
                cropped_image = translated_image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                cropped_mask = translated_mask.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

                # 4. Adjust annotations for cropping
                cropped_data = self.adjust_annotations_for_cropping(translated_data.copy(), crop_x_min, crop_y_min)

                # Save cropped image(original)
                new_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}.png"
                cropped_image.save(os.path.join(self.output_dir, new_filename))

                # Save cropped image(mask)
                new_mask_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}_mask.png"
                cropped_mask.save(os.path.join(self.output_dir, new_mask_filename))

                # Save JSON
                new_json_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_{i}_{j}.json"
                with open(os.path.join(self.output_dir, new_json_filename), 'w') as f:
                    json.dump(cropped_data, f, indent=4)
                
                # Save vis image
                self.visualization(cropped_data,new_filename)

    def adjust_annotations(self, data, tx, ty, ratio):
        """
        Adjusts the annotations based on translation and resize.

        Args:
            data (dict): Dictionary containing the image annotations.
            tx (int): Translation in x-axis.
            ty (int): Translation in y-axis.
            ratio (float): Resize ratio.

        Returns:
            dict: Dictionary containing the adjusted annotations.
        """
        data['box'][0] = (data['box'][0] + tx) * ratio
        data['box'][1] = (data['box'][1] + ty) * ratio
        data['box'][2] = (data['box'][2] + tx) * ratio
        data['box'][3] = (data['box'][3] + ty) * ratio

        data['Cx'] = (data['Cx'] + tx) * ratio
        data['Cy'] = (data['Cy'] + ty) * ratio

        data['dx'] = (data['dx']) * ratio
        data['dy'] = (data['dy']) * ratio
        data['R'] = (data['R'])* ratio

        data['w'] = (data['w']) * ratio
        data['h'] = (data['h']) * ratio

        return data
    
    def adjust_annotations_for_cropping(self, data, crop_x_min, crop_y_min):
        """
        Adjusts the annotations for cropping.

        Args:
            data (dict): Dictionary containing the image annotations.
            crop_x_min (int): Starting x-coordinate of the cropping region.
            crop_y_min (int): Starting y-coordinate of the cropping region.

        Returns:
            dict: Dictionary containing the adjusted annotations.
        """
        data['box'][0] -= crop_x_min
        data['box'][1] -= crop_y_min
        data['box'][2] -= crop_x_min
        data['box'][3] -= crop_y_min

        data['Cx'] -= crop_x_min
        data['Cy'] -= crop_y_min
        
        return data

if __name__ == "__main__":
    input_dir = r'E:\realman-robot\open_door\data\lever_handle'
    output_dir = r'E:\realman-robot\open_door\data\lever_handle_aug'
    augmentator = DataAugmentator(input_dir, output_dir)
    augmentator.augment_data()