"""Creates dataset with selected classes from
yolo dataset"""

import os
import shutil

# Get the list of all .txt files in the current directory.
path_names = 'D:\\PatoUTN\\Datasets\\originales\\2\\subimages'
path_new = 'D:\\PatoUTN\\Datasets\\originales\\Altered\\subimages'

total = 0
left = 0

class_indexes = [1] # Class indexes you want to KEEP

for split in ['train', 'val', 'test']:
    total = 0
    left = 0
    path_names_split = os.path.join(path_names, split)
    path_new_split = os.path.join(path_new, split)

    if not os.path.exists(path_new_split):
        os.makedirs(path_new_split, exist_ok=True)

    txt_files = [f for f in os.listdir(path_names_split) if f.endswith('.txt')]
    # Loop over each .txt file.
    for txt_file in txt_files:
        contains_objects = False
        total += 1
        # Open the .txt file in read mode.
        with open(os.path.join(path_names_split, txt_file), 'r') as f:        
            # Read the contents of the .txt file.
            lines = f.readlines()

            # Loop over each line in the .txt file.
            for line in lines:

                # Split the line into a list of words.
                words = line.split()            

                # Get the class index from the first word.
                current_index = words[0]
                
                for i in class_indexes:
                    if current_index == str(i):
                        contains_objects = True
                        with open(os.path.join(path_new_split, txt_file), 'a+') as fn:
                            fn.write(line)
                        fn.close()
        f.close()

        if contains_objects:
            left += 1
            img_file = txt_file.split('.')[0] + '.png'
            shutil.copyfile(os.path.join(path_names_split, img_file), os.path.join(path_new_split, img_file))

    print('Total images in', split, ':', total)
    print('Remaining images in', split, ':', left)