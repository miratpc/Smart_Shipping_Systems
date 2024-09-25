import math
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2

# Tools used for the inference code


def undisort_image(image_path, camera_data, pixel_coords, horizontal_distance, lateral_distance):
    # Load the image
    image = cv2.imread(image_path)
    
    # Load the camera data
    matrices = np.load(camera_data["cal_path"])
    intrinsic_matrix = matrices["mtx"]
    dist_coeffs = matrices["dist"]
    
    # Undistort the image
    undistorted_image = cv2.undistort(image, intrinsic_matrix, dist_coeffs)
    undistorted_image_rgb = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
    
    # Draw lines
    image_height, image_width, _ = undistorted_image.shape
    bottom_middle = (image_width // 2, image_height - 1)
    horz_point = (bottom_middle[0], pixel_coords[1])
    cv2.line(undistorted_image_rgb, bottom_middle, horz_point, (255, 0, 0), 2)
    cv2.line(undistorted_image_rgb, horz_point, pixel_coords, (0, 255, 0), 2)
    
    # Show the undistorted image
    plt.imshow(undistorted_image_rgb)
    
    
    plt.scatter(bottom_middle[0], pixel_coords[1], color='red', marker='x')
    plt.text(bottom_middle[0], pixel_coords[1], f"dist: {horizontal_distance:.2f}m", color='darkred')
    
    plt.scatter(pixel_coords[0], pixel_coords[1], color='lime', marker='x')
    plt.text(pixel_coords[0], pixel_coords[1], f"dist: {lateral_distance:.2f}m", color='darkgreen')
    
    plt.title("Undistorted Image")
    plt.axis("off")
    
    plt.show()
    
    return undistorted_image

def calculate_sensor_sizes(diagonal_mm, aspect_ratio: tuple):
    """ aspect_ratio: (width, height) 
    """
    mm_width = aspect_ratio[0] / np.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2) * diagonal_mm
    mm_height = aspect_ratio[1] / np.sqrt(aspect_ratio[0]**2 + aspect_ratio[1]**2) * diagonal_mm
    
    return mm_width, mm_height
    
def focal_length_in_mm(intrinsic_matrix, sensor_width, sensor_height, image_width, image_height):
    fl_x_pixels = intrinsic_matrix[0][0]  # horizontal focal length
    fl_y_pixels = intrinsic_matrix[1][1]  # vertical focal length
    
    fl_x_mm = fl_x_pixels * sensor_width / image_width
    fl_y_mm = fl_y_pixels * sensor_height / image_height
    #print("fl_x_mm: ",fl_x_mm)
    #print("fl_y_mm: ",fl_y_mm)
    return fl_x_mm, fl_y_mm

def calculate_pixel_angle(val, center_pixel, focal_length, mm_per_pixel):
    pixel_angle = math.atan((val - center_pixel) * mm_per_pixel / focal_length)
    return pixel_angle

def calculate_neighbour(opposite, pixel_angle):
    """
    Calculate the length of the adjacent side of a right triangle
    """
    
    return np.abs(opposite / np.tan(pixel_angle) if pixel_angle != 0 else np.inf)

def calculate_opposite(neighbour, pixel_angle):
    """
    Calculate the length of the opposite side of a right triangle
    """
    
    return np.abs(neighbour * np.tan(pixel_angle))

def calculate_distance_from_pixel(pixel_coords, camera_height, camera_data):
    cal_path = camera_data["cal_path"]
    
    try:
        matrices = np.load(cal_path)
        intrinsic_matrix = matrices["mtx"]
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None
    
    intrinsic_matrix = np.load(camera_data["cal_path"])["mtx"]
    img_width = int(camera_data["img_width"])
    img_height = int(camera_data["img_height"])
    aspect_ratio = (int(camera_data["aspect_ratio_x"]), int(camera_data["aspect_ratio_y"]))
    
    if "sensor_width" in camera_data and "sensor_height" in camera_data:
        sensor_width_mm = camera_data["sensor_width"]
        sensor_height_mm = camera_data["sensor_height"]
    else:
        diagonal_mm = float(camera_data["sensor_diagonal"])
        sensor_width_mm, sensor_height_mm = calculate_sensor_sizes(diagonal_mm, aspect_ratio)
    mmpp_x = sensor_width_mm / img_width
    mmpp_y = sensor_height_mm / img_height
    
    focal_length_x, focal_length_y = focal_length_in_mm(
        intrinsic_matrix,
        sensor_width_mm,
        sensor_height_mm,
        img_width,
        img_height

    )
    # print("focal_length_x: ",focal_length_x)
    # print("focal_length_y: ",focal_length_y)
    
    center_pixel_x = img_width / 2
    center_pixel_y = img_height / 2
    
    pixel_angle_x = calculate_pixel_angle(pixel_coords[0], center_pixel_x, focal_length_x, mmpp_x)
    pixel_angle_y = calculate_pixel_angle(pixel_coords[1], center_pixel_y, focal_length_y, mmpp_y)
    # print("---- pixel coords: ",pixel_coords)
    distance_ground = calculate_neighbour(camera_height, pixel_angle_y)
    distance_lateral = calculate_opposite(distance_ground, pixel_angle_x)
    total_distance = np.sqrt(distance_ground**2 + distance_lateral**2)
    
    
    return total_distance

