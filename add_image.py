import os
import shutil

goal_path = "D:\HCMAI_backup\src\app\static\image"  # Set your destination folder
start_path = "C:\Users\ASUS\Downloads\output_frames"  # Set your source folder

# Iterate through directories and move images to the goal directory
for video in os.listdir(start_path):
    video_path = os.path.join(start_path, video)
    for video_id in os.listdir(video_path):
        video_id_path = os.path.join(video_path, video_id)
        for image in os.listdir(video_id_path):
            image_path = os.path.join(video_id_path, image)
            target_path = os.path.join(goal_path, video, video_id)
            
            # Move the image to the target path
            shutil.move(image_path, os.path.join(target_path, image))
