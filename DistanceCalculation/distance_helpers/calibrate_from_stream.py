import cv2
import numpy as np



result_dir = r'DistanceCalculation/camera_cs30/calib2.npz'

# Defining the dimensions of the checkerboard
CHECKERBOARD = (4, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vectors to store vectors of 3D points for each checkerboard image
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Defining the world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Initialize video capture (use 0 for the default camera or provide a file path for a video)
cap = cv2.VideoCapture(0)  # Change 0 to your video file path if needed

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps // 4)  # Process 4 frames per second
frame_count = 0
num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        break
    
    frame_count += 1

    # Only process every `frame_interval` frame to achieve 4 FPS
    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine the pixel coordinates for 2D points
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
            num += 1
        
        # Display the frame with chessboard detection
        cv2.imshow('Calibration', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# Perform camera calibration after capturing enough frames
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Print calibration results
    print("Camera matrix: \n", mtx)
    print("Distortion coefficients: \n", dist)
    print("Rotation vectors: \n", rvecs)
    print("Translation vectors: \n", tvecs)
    
    # Save the calibration results
    np.savez(result_dir, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("No valid chessboard patterns found in the video.")
