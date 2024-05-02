import os
import pandas as pd
import numpy as np
import random
import subprocess
import shutil
import cv2

def copy_create(data_dataset_dir, split_name, name, i, data_dir_csv, data_dir_frame):
    temp = ["csv","frame","video"]

    match_folder = os.path.join(os.getcwd(), data_dataset_dir, split_name, f"{i}")
    if not os.path.exists(match_folder):
        os.makedirs(match_folder)
        print(f"create {match_folder}")
    for j in temp:
        j = os.path.join(match_folder, j)
        if not os.path.exists(j):
            os.makedirs(j)

        if j.endswith("csv"):
            source_file = os.path.join(data_dir_csv, f"{name}.csv")
            destination_file = os.path.join(j, f"{i}.csv")
            shutil.copy(source_file, destination_file)
        if j.endswith("frame"):
            frame_video_path  = os.path.join(data_dir_frame, name)
            
            image_name = os.listdir(frame_video_path)
            def extract_number(file_name):
                # Tách các phần số từ tên tệp
                number_str = ''.join(filter(str.isdigit, file_name))
                # Chuyển đổi chuỗi số thành số nguyên
                return int(number_str)

                # Sắp xếp danh sách các tên tệp ảnh bằng cách sử dụng hàm tùy chỉnh
            sorted_file_names = sorted(os.listdir(frame_video_path), key=extract_number)
            count = 0
            for img_name in sorted_file_names:
                source_file = os.path.join(os.getcwd(), frame_video_path, f"{img_name}")
                destination_folder = os.path.join(j, f"{i}")
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                    print(f"create {destination_folder}")
                destination_file = os.path.join(destination_folder, f"{count}.jpg")
                shutil.copy(source_file, destination_file)
                count = count + 1
    return

def create_video(data_dataset_dir, split, name, i):
    path_video = os.path.join(data_dataset_dir, split, f"{i}", "video", f"{i}.mp4")
    image_folder = os.path.join(data_dataset_dir, split, f"{i}", "frame", f"{i}")
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Sắp xếp các tệp tin ảnh theo thứ tự
    def extract_number(filename):
        # Tách các phần số từ tên tệp tin
        number_str = ''.join(filter(str.isdigit, filename))
        # Chuyển đổi chuỗi số thành số nguyên
        return int(number_str)
    images = sorted(images, key=extract_number)
    for i, img in enumerate(images):
        old_filename = os.path.join(image_folder, img)
        new_filename = os.path.join(image_folder, f"{i}.jpg")
        os.rename(old_filename, new_filename)
    new_images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    new_images = sorted(new_images, key=extract_number)

    # Đọc chiều rộng và chiều cao của ảnh đầu tiên để cấu hình video
    frame = cv2.imread(os.path.join(image_folder, new_images[0]))
    height, width, layers = frame.shape

    # Tạo đối tượng VideoWriter
    video = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    # Ghi các ảnh vào video
    for image in new_images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Đóng đối tượng VideoWriter
    cv2.destroyAllWindows()
    video.release()

    print("Video đã được tạo thành công!")
    print(path_video)
    return

def main():
    root = os.getcwd()
    data_dir_csv = 'data/GolfBall/Golf_Ball/Tracking/AnnotationsCsv'
    data_dir_frame = 'data/GolfBall/Golf_Ball/Tracking/JPEGImages'
    data_dataset_dir = 'data/GolfBall/Golf_Ball/Golf_Ball_Tracking_1'
    split = ["train","test"]

    name_video = []

    if not os.path.exists(data_dataset_dir):
        os.makedirs(data_dataset_dir)
        print(f"create {data_dataset_dir}")
        
    for i in os.listdir(data_dir_csv):
        i = i.split(".")[0]
        name_video.append(i)
    print(name_video)

    len_split = int(len(name_video) - np.floor(len(name_video)/7))
    train_dir = name_video[0: len_split]
    val_dir = name_video[len_split:]

    i = 0
    for name in train_dir:
        copy_create(data_dataset_dir, split[0], name, i, data_dir_csv, data_dir_frame)
        create_video(data_dataset_dir, split[0], name, i)
        i = i+1


    for name in val_dir:
        copy_create(data_dataset_dir, split[1], name, i, data_dir_csv, data_dir_frame)
        create_video(data_dataset_dir, split[1], name, i)
        i = i+1
    return 
if __name__ == "__main__":
    main()