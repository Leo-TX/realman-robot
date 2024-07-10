from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def add_point_to_image(image_path, x, y, dot_size=1, dot_color=(255, 0, 0),save_path=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Calculate the coordinates for the bounding box of the dot
    x1 = x - dot_size
    y1 = y - dot_size
    x2 = x + dot_size
    y2 = y + dot_size
    # Draw the dot on the image
    draw.ellipse((x1, y1, x2, y2), fill=dot_color)
    # Save the modified image
    if save_path:
        image.save(save_path)

def visualize_rotation(image_path, x1, y1, x2, y2, Ox, Oy, R, orientation='horizontal', angle=90, dot_size=3, dot_color=(255, 0, 0), circle_color=(0, 255, 0), arc_color=(0, 0, 255), save_path=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Add points
    add_point_to_image(image, x1, y1, dot_size, dot_color)
    add_point_to_image(image, x2, y2, dot_size, dot_color)
    # Draw the circle
    draw.ellipse((Ox-R, Oy-R, Ox+R, Oy+R), outline=circle_color)
    # Draw the arc
    angle_rad = np.radians(angle)
    if R > 0:
        start_angle = np.degrees(np.arctan2(y - Oy, x - Ox))
        end_angle = start_angle + angle
    else:
        end_angle = np.degrees(np.arctan2(y - Oy, x - Ox))
        start_angle = end_angle - angle
    draw.arc((Ox-R, Oy-R, Ox+R, Oy+R), start_angle, end_angle, fill=arc_color, width=2)
    # Display the visualized image using matplotlib for better quality
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()
    # Save the image if save_path is provided
    if save_path:
        image.save(save_path)
