import os
import pandas as pd
import numpy as np
import glob
import shutil   

# gpu=True
# gpu_number="0"
# if gpu:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# CHANGE SPLIT PATH FORM 3 TO 4 IF THERES NO TRAIN/VAL PATH
#set directory and savepath
# data_dir="ijcnn_v7data/high_density_100k/val"
data_dir="../../../data/Koutsoubn8/ijcnn_v7data/Real_world_test" #data directory
save_path="rw_test.csv"                     #location to save csv with area info
# size_limit= 32*32                          #max size to filter bbox
size_limit= 12*12 
# print(str(size_limit))
# exit()

w=480                                         #width for unorm
h=480                                         #height for unorm
save_directory = "eigen_out"
#create lists   
txts=[]
labels=[]
gen_data=False

 # 10 imges from 0-200, 200-400,
lower_limit=0
upper_limit=100000
#create direcotries if not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    # os.makedirs(save_directory + "/images/")    
    os.makedirs(save_directory +"/labels/")

# exit()
#walk through directory and append all the label.txts into list
for root, dirs, files in os.walk(data_dir + "/labels/",topdown=True):
    for file in files:
        if file.endswith(".txt"):
            txts.append(os.path.join(root,file))

# read each label file line by line to gather all labels
for i, txt in enumerate(txts):
    with open(txt) as txt:       
        for line in txt:
            labels.append(line)

#Create a dataframe iwth all the file paths
txtdf=pd.DataFrame(txts,columns=["name"])

#create a column for the file path
path_name=txtdf["name"]
path_name=pd.DataFrame(path_name)
path_name.rename({'name':'path'},inplace=True,axis='columns')

#create a column with just the file name
split_path=txtdf['name'].str.split(pat=("/"),expand=True)
# print(split_path)
# print("split_path",split_path)
# exit()
print("splt",split_path)
# exit()
file_names=pd.DataFrame(split_path[8].str.split(pat=".",expand=True))

file_names.rename({0:'name'},inplace=True,axis="columns")

#combine into one dataframe
label_info=pd.concat([path_name,file_names['name']],axis=1)

#create dataframe with label information and split into cxywh
labeldf=pd.DataFrame(labels,columns=["label"])
labeldf=labeldf["label"].str.split(" ",expand=True)
labeldf.rename({0:"class",1:'x',2:'y',3:'w',4:'h'},axis="columns",inplace=True)

#append to label info
label_info=pd.concat([label_info,labeldf],axis=1)
# print("label_info ",label_info)
# exit()

#gather the widths and heights, convert to float and unormoalize
width=label_info["w"].astype("float32")*w
height=label_info["h"].astype('float32')*h

#get the area for the ground truths
area=width*height
area=pd.DataFrame(area)
area.rename({0:'area'},axis=1,inplace=True)
# print(area)
# exit()
label_info=pd.concat([label_info,area],axis=1)

#not important
area=area.dropna()
print(area)
avg=area.mean()
absolute_sizes=abs(area-avg)
print(absolute_sizes)
avg_abs_size=(absolute_sizes.mean())
print("C", avg_abs_size)
# print(np.sqrt(avg_abs_size))



# exit()
#sort values by area
label_info=label_info.sort_values(by=["area"])
#obtain the small bounding boxes to add to csv
sizes=label_info['area'] < upper_limit
sizes=pd.DataFrame(sizes)
sizes.rename({'area':'small'},axis=1,inplace=True)
# label_info=pd.concat([label_info,sizes],axis=1)

# print(type(label_info["area"]))
#create a mask for the small objects and apply

less_mask=(label_info["area"] < upper_limit)
greater_mask=(label_info["area"] > lower_limit)

less_objs=label_info[less_mask]
greater_obj=label_info[greater_mask]
# print(less_mask)
# print(greater_mask)
# exit()
filtered_objs=label_info[less_mask & greater_mask]
# print(filtered_objs)
# exit()
filtered_objs=filtered_objs.sample(frac=1,random_state=1).reset_index(drop=True)
#drop out nan names
labels=filtered_objs["name"].dropna()
#gather the images corresponding to the small labels

# label_info.to_csv(save_path)
sm_img=[]
sm_lb=[]
# print(filtered_objs)
#shuffle dataframe to ensure variety of data
# labels=labels.sample(frac=1,random_state=1).reset_index(drop=True)
# shuf_area=label_info["area"].sample(frac=1,random_state=1).reset_index(drop=True)
# print(s)
# exit()
# print(len(small_labels))
# exit()
if gen_data:
    for i,files in enumerate(labels[0:3]):
        print("files",files)
        data_area=int(filtered_objs['area'][i])
        imgs=(glob.glob(data_dir + "/images/" + files +".jpg"))
        labels=(glob.glob(data_dir + "/labels/" + files +".txt"))

        print("imgs",imgs)
        print("labels",labels)
        # sm_lb.append(small_labels)
        shutil.copy(imgs[0], save_directory)
        shutil.copy(labels[0], save_directory)
        print("images and labels copied to ", save_directory, ": ",files, " ",i)
        # print(save_directory + "/images/" + files + ".jpg")
        # try:
        #     os.rename(save_directory + "/images/" + files +".jpg", save_directory  +"/images/" +files+"_area_"+ str(data_area) +".jpg")
        #     os.rename(save_directory + "/labels/" + files +".txt", save_directory  +"/labels/" +files+"_area_" + str(data_area) +".txt")
        # except:
            # print("improper name")
        # os.rename(labels[0], save_directory + "/labels/")
    
        # print("failed to generate image")
    
# sm_img.sort()
# sm_lb.sort()
# print(sm_img)

# priorlist=pd.read_csv(save_path)
# filtered_objs=priorlist.append(filtered_objs,ignore_index=True)
# filtered_objs=pd.concat([filtered_objs,priorlist[:10]],axis=0,ignore_index=True)
filtered_objs=filtered_objs.sort_values(by=['area']).reset_index(drop=True)
# filtered_objs=pd.concat([filtered_objs,priorlist],axis=0)
filtered_objs.to_csv(save_path)
print(filtered_objs[:4])
names=filtered_objs['name']
print(data_dir + "/labels/" + names[0] +".txt")
print(data_dir + "/images/" + names[0] +".jpg")
