# This code will convert MAVRC format to YOLOv5/YOLOv7 format

# data_path folder should contain folders with the following format:
#
# data_path_dir
# --- Video1
# ------ frame1.jpg
# ------ ...
# ------ labels.csv
# --- ...

# out_path will be in the format of 
# out_path_dir
# --- images
# ------ frame1.jpg
# ------ ...
# --- labels
# ------ frame1.txt
# ------ ...

# Imports
import os
import pandas as pd
import cv2
import argparse

# Settings
#--------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to data folder')
parser.add_argument('--save_path', type=str, help='path to save folder in yolo format')
paths = parser.parse_args()

data_path = paths.data_path + "/"
out_path  = paths.save_path + "/"

# 
if not os.path.exists(out_path):
    os.makedirs(out_path)

if not os.path.exists(out_path + "images/"):
    os.makedirs(out_path + "images/")

if not os.path.exists(out_path + "labels/"):
    os.makedirs(out_path + "labels/")


def center_xy_coords(xywh):
    x,y,w,h = xywh
    center_x, center_y = (x+x+w)/2,(y+y+h)/2
    return [center_x, center_y,w,h]


def normalize_xywh(xywh, w, h):
    print(w,h)
    print("unnorm",xywh)
    xywh[0] = xywh[0] / w
    xywh[1] = xywh[1] / h
    xywh[2] = xywh[2] / w
    xywh[3] = xywh[3] / h
    print("norm",xywh)
    return xywh

# Load data
#--------------------------------------------------------------------#
category_ids = ["AIRPLANE", "DRONE"]


dir_list = [x[0] for x in os.walk(data_path)]

for sub_dir in dir_list:
    sub_dir_list = os.listdir(sub_dir)
   
    mp4_file = [ filename for filename in sub_dir_list if filename.endswith( ".mp4" ) ]
    csv_file = [ filename for filename in sub_dir_list if filename.endswith( ".csv" ) ]
   
    if len(mp4_file) == 1 and len(csv_file) == 1 and "Bounding Boxes" not in sub_dir:
        
       
        mp4_file = mp4_file[0]
        csv_file = csv_file[0]
        sub_dir_save_name = "".join(["_" if x == " " or x == "/" else x for x in sub_dir.replace(data_path,'')]) # Remove data_path, spaces and "/"
        
        # Load data
        video_labels = pd.read_csv(sub_dir + "/" + csv_file)
       
        
        # Load Video
        video_capture = cv2.VideoCapture(sub_dir + "/" + mp4_file)
        assert video_capture.isOpened(), "Error opening " + sub_dir + "/" + mp4_file

        # Collect data from video
        # fps             = int(video_capture.get(cv2.CAP_PROP_FPS          ))
        num_frames      = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT  ))
        # width, height =480,480
        for frame_num in range(num_frames):

            # Read next frame video_capture
            is_frame_returned, frame = video_capture.read()

            # Check if frame is returned
            if is_frame_returned:
                if frame_num == 0:
                    print(sub_dir)
                
                # Save image
                frame= cv2.resize(frame,(480,480))
                # print(frame.size)
                # exit()
               
                cv2.imwrite(out_path + "images/" + sub_dir_save_name + "_" + str(frame_num) + ".jpg", frame)
            
                # 
                frame_labels = video_labels.loc[video_labels.index == frame_num]
                print(frame_labels)
                # 
                with open(out_path + "labels/" + sub_dir_save_name + "_" + str(frame_num) + ".txt", 'a') as txt_file:
                    for _, row in frame_labels.iterrows():
                        class_name = frame_labels.columns[2]
                        if not pd.isna(class_name):
                            print(row)
                            # exit()
                            print(frame_labels
                            exit())
                            label = str(category_ids.index(frame_labels.columns[3]))
                            print("label",label)

                            try:
                                xywh = eval(row.AIRPLANE)
                            except:
                                pass
                            try:
                                xywh =eval(row.DRONE)
                            except:
                                pass

                            print(xywh)
                            # xywh=xywh2xyxy(xywh)
                            xywh=center_xy_coords(xywh)
                            print(xywh)
                            norm_xywh = normalize_xywh(xywh, 1920, 1080)
                            # print(norm_xywh)
                        
                        
                            # Write to txt
                            txt_file.write(label + ' ')
                            for dim_num, dim in enumerate(xywh):
                                print("dim",dim)
                            
                                if dim_num == 3:
                                    txt_file.write(str(dim))
                                else:
                                    txt_file.write(str(dim) + ' ')
                            txt_file.write('\n')
                            # exit()
                          
                            
                            

