import os
import cv2
import yaml
from ultralytics import YOLO
from tqdm import tqdm
from distance_tools import calculate_distance_from_pixel, undisort_image

# Configuration parameters
CONFIG = {
    'camera_height': 2.06,  # in meters
    'iou_threshold': 0.4, # Intersection over Union threshold for filtering out duplicate boxes
    'threshold': 0.15,  # detection threshold for the model
    'show_boxes': True, # Show bounding boxes on the output
    'show_horizon': False, # Show the horizon line on the output
}


PATHS = {
    'model_path': [r'best_models\Old_Ft_ft.pt',
                   ],
    
    
    
    'videos_dir': [r"path\to\videos\dir",
                   ],
    
    'yaml_path': r'DistanceCalculation\camera_cs30\camera.yaml',
}


EXTENTIONS = {
    'video_extension': (".mp4", ".MP4", ".avi", ".AVI"),
    'image_extension': (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"),
}





# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2, _, _ = box1
    x1b, y1b, x2b, y2b, _, _ = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# Function to filter out duplicate bounding boxes
def filter_boxes(boxes, iou_threshold=CONFIG['iou_threshold']):
    filtered_boxes = []
    for i in range(len(boxes)):
        keep = True
        for j in range(len(filtered_boxes)):
            if calculate_iou(boxes[i], filtered_boxes[j]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered_boxes.append(boxes[i])
    return filtered_boxes

def load_camera_data(yaml_path=PATHS['yaml_path']):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            if 'camera' not in data:
                raise KeyError("The 'camera' key is missing from the YAML file.")
            return data['camera']
    except FileNotFoundError:
        print(f"Error: The file at {yaml_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def process_video(video_path, video_path_out, model, cnfg, camera_data=None):
    
    threshold=cnfg['threshold']
    iou_threshold=cnfg['iou_threshold']
    
    show_boxes=cnfg['show_boxes']
    show_horizon=cnfg['show_horizon']
    

    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return




    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    middle_point = (frame_width // 2, frame_height // 2)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, fps, (frame_width, frame_height))

    if camera_data is None:
        # Load camera data from YAML file
        camera_data = load_camera_data()
    
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False, conf=threshold)[0]
        boxes = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))

        boxes = filter_boxes(boxes, iou_threshold = iou_threshold)
        # print("boxes: ",boxes)

        if boxes:
            if show_horizon:
                cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 0, 255), 2)
                
            camera_height = cnfg['camera_height']
            
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                
                bottom_most_point = (x1 + x2) // 2, y2 # middle of the bottom edge
                
                """ if x2 < middle_point[0]: # closest point to the middle
                    bottom_most_point = (x2, y2)
                elif x1< middle_point[0] < x2:
                    bottom_most_point = (middle_point[0], y2)
                else:
                    bottom_most_point = (x1, y2) """
                
                
                if camera_data:
                    distance_ground = calculate_distance_from_pixel(bottom_most_point, camera_height, camera_data)
                    
                    if show_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)  # Draw the bounding box in soft red
                    # Annotate the frame
                    cv2.circle(frame, bottom_most_point, 5, (0, 0, 255), -1)  # Mark the point
                    cv2.putText(frame, f"{distance_ground:.2f}m", (bottom_most_point[0] + 10, bottom_most_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    print("Camera data could not be loaded.")
        
        
        pbar.update(1)
        out.write(frame)

    cap.release()
    out.release()

def process_image(image_path, image_path_out, model, cnfg):
    
    threshold=cnfg['threshold']
    iou_threshold=cnfg['iou_threshold']
    
    show_boxes=cnfg['show_boxes']
    show_horizon=cnfg['show_horizon']
    
    
    
    
    image = cv2.imread(image_path)
    middle_point = (image.shape[1] // 2, image.shape[0] // 2)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    results = model(image, verbose=False, conf=threshold)[0]
    boxes = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))

    boxes = filter_boxes(boxes, iou_threshold = iou_threshold)
    print("boxes: ",boxes)
    if boxes:
        if show_horizon:
            cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (0, 0, 255), 2)
            
        camera_height = cnfg['camera_height']
        
        # Assuming your YAML file and calibration data is available
        camera_data = load_camera_data()
        
        if camera_data:
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                bottom_most_point = (x1 + x2) // 2, y2 # middle of the bottom edge
                
                """ if x2 < middle_point[0]:
                    bottom_most_point = (x2, y2)
                elif x1< middle_point[0] < x2:
                    bottom_most_point = (middle_point[0], y2)
                else:
                    bottom_most_point = (x1, y2) """
                
                # Calculate the distance for the current bounding box
                distance_ground = calculate_distance_from_pixel(bottom_most_point, camera_height, camera_data)
                if show_boxes:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 2)  # Draw the bounding box in soft red
                    
                # Annotate the image
                cv2.circle(image, bottom_most_point, 5, (0, 0, 255), -1)  # Mark the point
                cv2.putText(image, f"{distance_ground:.2f}m", (bottom_most_point[0] + 10, bottom_most_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                
            # Save the processed image
            cv2.imwrite(image_path_out, image)
        else:
            print("Camera data could not be loaded.")
    else:
        print("No valid bounding boxes found.")







if __name__ == "__main__":
    # Define the directories and file paths using CONFIG
        
    total_pbar = tqdm(total=len(PATHS['videos_dir']) * len(PATHS['model_path']), desc="Total progress", position=0, leave=True)

    for dir in PATHS['videos_dir']:
        images_list = []
        videos_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                fname, ext = os.path.splitext(file)
                if "_annotated" in fname:
                    continue
                if ext in EXTENTIONS['video_extension']:
                    videos_list.append(os.path.join(root, file))
                elif ext in EXTENTIONS['image_extension']:
                    images_list.append(os.path.join(root, file))
        
        for model_path in PATHS['model_path']:
            
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model = YOLO(model_path, verbose=False)
            
            for img_path in tqdm(images_list, desc="Processing images", position=1, leave=False):
                root, file = os.path.split(img_path)
                stripped_file, extention = os.path.splitext(file)
                out_dir = os.path.join(root, f"{model_name}_output")
                os.makedirs(out_dir, exist_ok=True)
                image_path_out = os.path.join(out_dir, stripped_file + "_annotated" + extention)
                process_image(img_path, image_path_out, model, CONFIG)
            
            for video_pth in tqdm(videos_list, desc="Processing videos", position=1, leave=False):
                root, file = os.path.split(video_pth)
                stripped_file, extention = os.path.splitext(file)
                out_dir = os.path.join(root, f"{model_name}_output")
                os.makedirs(out_dir, exist_ok=True)
                video_path_out = os.path.join(out_dir, stripped_file + "_annotated" + extention)
                process_video(video_pth, video_path_out, model, CONFIG)

            total_pbar.update(1)
    
    total_pbar.close()
    print("All videos and images have been processed.")