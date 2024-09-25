import cv2
import os
# RTSP stream URL
rtsp_url = "rtsp://"
save_path = r"screenshot_folder"


# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

# Check if the capture is successful
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
num = 0
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow("Video Stream", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite( os.path.join(save_path, f"\photo_{num}.jpg", frame))
        num += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()