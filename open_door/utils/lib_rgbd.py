from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2

def rotate_point(x1_2d,y1_2d,R,orientation='horizontal',angle=90):
        angle_rad = np.radians(angle)
        if orientation == 'horizontal':
            Ox = x1_2d + R
            Oy = y1_2d
            if R > 0:
                angle_rad *= -1
        elif orientation =='vertical':
            Ox = x1_2d
            Oy = y1_2d + R
            if R < 0:
                angle_rad *= -1
        x1_2d -= Ox
        y1_2d -= Oy
        x2_2d = x1_2d * np.cos(angle_rad) - y1_2d * np.sin(angle_rad)
        y2_2d = x1_2d * np.sin(angle_rad) + y1_2d * np.cos(angle_rad)
        x2_2d += Ox
        y2_2d += Oy
        return x2_2d,y2_2d,Ox,Oy

def add_point_to_image(img, x, y, dot_size=1, dot_color=(255, 0, 0),save_path=None):
    if isinstance(img,str):
        image = Image.open(img).convert("RGB")
    else:
        image = img
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

def vis_grasp(img_path, dx, dy, x1, y1, x2, y2, Ox, Oy, R, orientation='horizontal', angle=90, save_path=None, show=False):
    # pattern
    dot_size=5
    dot_color=(255, 0, 0) # red
    circle_color=(0, 255, 0) # green
    arc_color=(0, 0, 255) # blue
    line_color = (255, 255, 0) # yellow
    circle_width=3
    arc_width=3

    plt.figure()
    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)

    # Add points
    add_point_to_image(image, x1-dx, y1-dy, dot_size, dot_color)
    add_point_to_image(image, x1, y1, dot_size, dot_color)
    add_point_to_image(image, x2, y2, dot_size, dot_color)
    add_point_to_image(image, Ox, Oy, dot_size, dot_color)
    
    # Draw lines from P1 and P2 to center
    draw.line((x1, y1, Ox, Oy), fill=line_color, width=2)
    draw.line((x2, y2, Ox, Oy), fill=line_color, width=2)

    x0 = min(Ox - R, Ox + R)
    y0 = min(Oy - R, Oy + R)
    x1 = max(Ox - R, Ox + R)
    y1 = max(Oy - R, Oy + R)
    
    # Draw the circle
    draw.ellipse((x0, y0, x1, y1), outline=circle_color, width=circle_width)
    
    # Draw the arc
    if orientation=='horizontal':
        if R<0:
            start_angle = 0
            end_angle = np.degrees(np.arctan2(y2 - Oy, x2 - Ox))
        else:
            start_angle = np.degrees(np.arctan2(y2 - Oy, x2 - Ox))
            end_angle = 180
    elif orientation=='vertical':
        if R<0:
            start_angle = np.degrees(np.arctan2(y2 - Oy, x2 - Ox))
            end_angle = 90
        else:
            start_angle = -90
            end_angle = np.degrees(np.arctan2(y2 - Oy, x2 - Ox))
    if end_angle < start_angle:
        end_angle += 360
    draw.arc((x0, y0, x1, y1), start_angle, end_angle, fill=arc_color, width=arc_width)

    # Display the visualized image using matplotlib for better quality
    plt.imshow(np.array(image), extent=[0, img_w, 0, img_h])
    plt.title("Visualized Grasp")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    if show:
        plt.show()
    # Save the image if save_path is provided
    if save_path:
        image.save(save_path)

def vis_rgbd(d_img_path, rgb_img_path=None, save_path="depth_visualization.png"):
    # Load the depth image
    depth_img = cv2.imread(d_img_path, cv2.IMREAD_UNCHANGED)
    # Create a mask where zero-depth pixels are True (white)
    zero_depth_mask = depth_img == 0
    # Create an empty image (initially black)
    visualized_depth = np.zeros_like(depth_img, dtype=np.uint8)
    # Set zero-depth pixels to white
    visualized_depth[zero_depth_mask] = 255
    # Optional: Blend with RGB image for better visualization
    if rgb_img_path is not None:
        rgb_img = cv2.imread(rgb_img_path)
        # Resize RGB image to match depth image dimensions if needed
        if rgb_img.shape[:2] != depth_img.shape[:2]:
            rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]))
        # ----> Convert depth image to 3 channels:
        visualized_depth = cv2.cvtColor(visualized_depth, cv2.COLOR_GRAY2BGR)
        # Blend images (adjust alpha for desired transparency)
        alpha = 0.5  # Example: 50% transparency
        visualized_depth = cv2.addWeighted(visualized_depth, alpha, rgb_img, 1 - alpha, 0)
    # Save the visualized depth image
    if save_path:
        cv2.imwrite(save_path, visualized_depth)
        # print(f"Depth visualization saved to: {save_path}")

def vis_d(d_img_path,save_path,show=False):
    # Load the depth image
    depth_image = cv2.imread(d_img_path, cv2.IMREAD_ANYDEPTH)
    # Check if the image is loaded correctly
    if depth_image is None:
        print("Error: Could not load the depth image.")
        exit()
    # Normalize the depth image to 0-255 range for visualization
    output_displayable = cv2.normalize(depth_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_displayable = cv2.cvtColor(output_displayable, cv2.COLOR_GRAY2BGR)
    # Display the depth image using matplotlib
    plt.imshow(output_displayable)
    plt.title("Depth Image")
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)  # Save the image
        # print(f"Depth visualization saved to: {save_path}") 
    if show:
        plt.show()

def crop_image(img_path, center_x, center_y, new_w, new_h, save_path=None):
    # Open the image
    image = Image.open(img_path)
    
    # Get the width and height of the image
    img_width, img_height = image.size
    
    # Calculate the coordinates of the top left and bottom right corners of the crop area
    left = center_x - new_w // 2
    top = center_y - new_h // 2
    right = left + new_w
    bottom = top + new_h
    
    # Handle special cases: adjust the crop area if it goes beyond the image boundaries
    if left < 0:
        left = 0
        right = new_w
    elif right > img_width:
        right = img_width
        left = img_width - new_w
    
    if top < 0:
        top = 0
        bottom = new_h
    elif bottom > img_height:
        bottom = img_height
        top = img_height - new_h
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # save
    if save_path:
        cropped_image.save(save_path)
    
    # Return the cropped image
    return cropped_image

if __name__ == "__main__":
    x1_2d,y1_2d = 368.35743484925905,427.1806336228922
    dx = -8
    dy = 50
    R = -150
    orientation = 'vertical'
    angle=90
    img_path = r'E:\realman-robot\open_door\data\trajectory_045\1\rgb.png'
    x2_2d,y2_2d,Ox,Oy = rotate_point(x1_2d,y1_2d,R,orientation,angle)
    vis_grasp(img_path, dx, dy,x1_2d,y1_2d,x2_2d,y2_2d,Ox,Oy,R,orientation,angle,save_path=img_path.replace('rgb','vis_grasp'),show=True)
