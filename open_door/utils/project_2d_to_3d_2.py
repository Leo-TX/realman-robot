import cv2
import numpy as np

def project_2d_to_3d_camera(x, y, intrinsic_matrix, dist_coeffs, depth=None):
  # Convert to numpy array
  image_point = np.array([x, y, 1], dtype=np.float32).reshape(3, 1)

  # Convert to normalized coordinates
  normalized_point = cv2.undistortPoints(image_point, intrinsic_matrix, dist_coeffs)

  if depth is not None:
    # Scale by depth if available
    point_3d_camera = normalized_point * depth
  else:
    # No depth information, obtain direction
    point_3d_camera = normalized_point 
    print("Warning: Depth information is missing, only direction is obtained.")

  return point_3d_camera

# Example usage
x, y = 100, 200  # Your 2D image point
intrinsic_matrix = np.array([[902.9446,0,637.6256],[0,903.3026,366.0768],[0,0,1]])
dist_coeffs = np.array([0.0898146905700043,-0.189559858564607,0,0,0])

# With depth information (example depth value)
depth = 1.5
point_3d = project_2d_to_3d_camera(x, y, intrinsic_matrix, dist_coeffs, depth)

# Without depth information
point_3d_direction = project_2d_to_3d_camera(x, y, intrinsic_matrix)