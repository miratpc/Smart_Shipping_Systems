import os
import cv2
import shutil
from tqdm import tqdm

#print how many frames are in the video
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def split_video(video_path, output_folder, video_name):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_name} has {length} frames")
    count = 0
    pbar = tqdm(total=length, leave=False, desc=f'{video_name}')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_folder, f'{video_name}_frame_{count:06d}.jpg'), frame)
            count += 1
            pbar.update(1)
        else:
            break
    cap.release()

    
if __name__ == '__main__':    
    videos_folder = r'path/to/videos'
    images_folder = r'path/to/images'
    labels_folder = r'path/to/labels'
    
    
    folder_names = os.listdir(videos_folder)
    print(folder_names)
    
    for video_name in folder_names:
        if video_name.startswith('.'):
            continue
        
        videoname_path = os.path.join(videos_folder, video_name)
        for temp in os.listdir(videoname_path):
            if temp.endswith('.mp4'):
                video_path = os.path.join(videoname_path, temp)
                break
        
        split_video(video_path, images_folder, video_name)
        
        labels_dir = os.path.join(videoname_path, 'labels')
        for i in os.listdir(labels_dir):
            if i.lower() == "train":
                labels_dir = os.path.join(labels_dir, i)
                break
            elif i.lower() == "test":
                labels_dir = os.path.join(labels_dir, i)
                break
            elif i.lower() == "val":
                labels_dir = os.path.join(labels_dir, i)
                break
            else:
                continue
        
        for frame in os.listdir(labels_dir):
            oldpth = os.path.join(labels_dir, frame)
            newpth = os.path.join(labels_folder, video_name + "_" + frame)
            
            shutil.move(oldpth, newpth)
        
        
        print(f'{video_name} done')
        
    
    print('All done')
    cv2.destroyAllWindows()
    