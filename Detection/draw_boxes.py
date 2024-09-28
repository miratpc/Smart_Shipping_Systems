import os
from ultralytics import YOLO
from tqdm import tqdm
import cv2



VIDEOS_DIR = r"Boat_Detection\videos"
model_path = r'runs\detect\2Class_GroundUp\weights\best.pt'
threshold = 0.15




# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2, _, _= box1
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
def filter_boxes(boxes, iou_threshold=0.7):
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



def process_video(video_path, video_path_out, model, threshold=0.15):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video.")
        cap.release()
        exit()

    # Get frame dimensions
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))


    pbar = tqdm(desc= video_name, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), unit="fr", leave=False)
    while ret:
        # Run the model on the frame
        results = model(frame, verbose=False, conf=threshold)[0] 

        # Collect bounding boxes and their scores
        boxes = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))

        # Filter out duplicate bounding boxes
        boxes = filter_boxes(boxes)

        # Draw the filtered bounding boxes on the frame
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() + ' ' + str(round(score, 2)),
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Write the frame with detections to the output video
        out.write(frame)
        pbar.update(1)
        # Read the next frame
        ret, frame = cap.read()

    # Release resources
    cap.release()
    out.release()

def process_image(image_path, image_path_out, model, threshold=0.15):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        exit()

    # Run the model on the image
    results = model(image, verbose=False, conf=threshold)[0]

    # Collect bounding boxes and their scores
    boxes = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))

    # Filter out duplicate bounding boxes
    boxes = filter_boxes(boxes)

    # Draw the filtered bounding boxes on the image
    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper() + ' ' + str(round(score, 2)),
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Save the annotated image
    cv2.imwrite(image_path_out, image)



if __name__ == "__main__":
    

    images_list = []
    videos_list = []
    for root, dirs, files in os.walk(VIDEOS_DIR):
        
        for file in files:
            fname, ext = os.path.splitext(file)
            if "_annotated" in fname:
                continue
            if ext in (".mp4", ".MP4"):
                videos_list.append(os.path.join(root, file))
            elif ext in (".jpg", ".JPG", "jpeg", "JPEG", ".png", ".PNG"):
                images_list.append(os.path.join(root, file))
                
                
    model = YOLO(model_path, verbose=False)  # Load a custom model
    
    for img_path in tqdm(images_list, desc="Processing images"):
        
        root, file = os.path.split(img_path)
        stripped_file, extention = os.path.splitext(file)
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        image_path_out = os.path.join(out_dir, stripped_file + "_annotated" + extention)
        
        process_image(img_path, image_path_out, model, threshold=threshold)
    
    for video_pth in tqdm(videos_list, desc="Processing videos"):
        
        root, file = os.path.split(video_pth)
        stripped_file, extention = os.path.splitext(file)
        out_dir = os.path.join(root, "output")
        os.makedirs(out_dir, exist_ok=True)
        video_path_out = os.path.join(out_dir, stripped_file + "_annotated" + extention)
    
        process_video(video_pth, video_path_out, model, threshold=threshold)