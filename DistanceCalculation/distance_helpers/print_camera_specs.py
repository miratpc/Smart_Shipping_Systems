import cv2

# Initialize video capture (use 0 for the default camera or provide a file path for a video)
cap = cv2.VideoCapture(0)  # Change 0 to your video file path if needed

ss_save_path = r"name_of_snapshot.jpg"

if not cap.isOpened():
    print("Error: Unable to open video stream or file.")
else:
    # Get the width, height, fps, and calculate aspect ratio
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    aspect_ratio = frame_width / frame_height

    # Additional properties (if supported by the device)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    codec = cap.get(cv2.CAP_PROP_FOURCC)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    focus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    # get a snapshot
    ret, frame = cap.read()
    cv2.imwrite(ss_save_path, frame)
    
    
    # Print the basic information
    print(f"Frame Width: {frame_width} pixels")
    print(f"Frame Height: {frame_height} pixels")
    print(f"Frames per Second (FPS): {fps}")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    # Print additional information
    print(f"Brightness: {brightness}")
    print(f"Contrast: {contrast}")
    print(f"Saturation: {saturation}")
    print(f"Hue: {hue}")
    print(f"Gain: {gain}")
    print(f"Exposure: {exposure}")
    print(f"Video Codec (FOURCC): {int(codec)}")
    print(f"Total Frames (if video file): {frame_count}")
    print(f"Autofocus Status: {focus}")

    # Release the video capture object
    
    cap.release()

cv2.destroyAllWindows()