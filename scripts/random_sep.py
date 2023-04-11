# Copyright from Sharebee.cn Inc All rights Reserved.
# Author: Samuel
# Date: April 10, 2023
# Reference: https://github.com/Samuel-wei/Alexey-darknet/blob/master/gen_files.py and GPT

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
import matplotlib
import shutil 


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def sep_train_val(file):
    name_extention = file.split('.')[-1]
    if probo < 75 :
        if name_extention == 'jpg':
            shutil.copy(file,train_img_dst_path)
        if name_extention == 'xml':
            shutil.copy(file,train_xml_dst_path)      
    else:
        if name_extention == 'jpg':
            shutil.copy(file,val_img_dst_path)
        if name_extention == 'xml':
            shutil.copy(file,val_xml_dst_path)    

  
# Check path and input parameters
#wd = os.getcwd()
wd = '/home/workspace/'
print(wd)

# work_space_dir = os.path.join(wd)
work_space_dir = os.path.join(wd, 'PCBBoard/')
if not os.path.isdir(work_space_dir):
    os.makedirs(work_space_dir)
image_dir = os.path.join(work_space_dir, "image_crop/")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
clear_hidden_files(image_dir)
annotation_dir = os.path.join(work_space_dir, "image_xml/")
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)
clear_hidden_files(annotation_dir)

train_file = open(os.path.join(work_space_dir, "train2017_1.txt"), 'w')
val_file = open(os.path.join(work_space_dir, "val2017_1.txt"), 'w')
train_file.close()
val_file.close()

train_img_dst_path = "/home/workspace/PCBBoard/images/train2017"
if not os.path.isdir(train_img_dst_path):
    os.makedirs(train_img_dst_path)
train_xml_dst_path = "/home/workspace/PCBBoard/xmls/train2017"
if not os.path.isdir(train_xml_dst_path):
    os.makedirs(train_xml_dst_path)

val_img_dst_path = '/home/workspace/PCBBoard/images/val2017'
if not os.path.isdir(val_img_dst_path):
    os.makedirs(val_img_dst_path)
val_xml_dst_path = '/home/workspace/PCBBoard/xmls/val2017'
if not os.path.isdir(val_xml_dst_path):
    os.makedirs(val_xml_dst_path)


train_file = open(os.path.join(work_space_dir, "train2017_1.txt"), 'a')
val_file = open(os.path.join(work_space_dir, "val2017_1.txt"), 'a')
	
list = os.listdir(image_dir) # list image files
probo = random.randint(1, 100)
print("Probobily: %d" % probo)
num = 0
for i in range(0, len(list)):   
    num += 1
    if num == 201:
        break
    path = os.path.join(image_dir, list[i])
    if os.path.isfile(path):
        image_path = image_dir + list[i]
        voc_path = list[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
    probo = random.randint(1, 100)
    print("probobility: %d" % probo)
    if (probo < 75):
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            sep_train_val(annotation_path)
            sep_train_val(image_path)
            #convert_annotation(nameWithoutExtention)
    else:
        if os.path.exists(annotation_path):
            val_file.write(image_path + '\n')
            sep_train_val(annotation_path)
            sep_train_val(image_path)
            #convert_annotation(nameWithoutExtention)
train_file.close()
val_file.close()


