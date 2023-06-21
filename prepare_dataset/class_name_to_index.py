import os

# Get the list of all .txt files in the current directory.
path_names = '/shared/PatoUTN/PAP/Datasets/originales/2/yolo/test'
path_new = '/shared/PatoUTN/PAP/Datasets/originales/2/yolo_idx_class/test'
txt_files = [f for f in os.listdir(path_names) if f.endswith('.txt')]

# Loop over each .txt file.
for txt_file in txt_files:

    # Open the .txt file in read mode.
    with open(os.path.join(path_names, txt_file), 'r') as f:
        with open(os.path.join(path_new, txt_file), 'w') as fn:

            # Read the contents of the .txt file.
            lines = f.readlines()

            # Loop over each line in the .txt file.
            for line in lines:

                # Split the line into a list of words.
                words = line.split()
                

                # Get the class name from the first word.
                class_name = words[0]

                # If the class name is "Normal", set the class index to 0.
                if class_name == "Normal":
                    class_index = 0

                # If the class name is "Altered", set the class index to 1.
                elif class_name == "Altered":
                    class_index = 1

                # Case for single class detection
                elif class_name == "Cell":
                    class_index = 0

                # Replace the class name with the class index in the line.
                words[0] = str(class_index)

                # Join the list of words into a single line.
                new_line = ' '.join(words)

                # Write the new line to the .txt file.
                
                fn.write(new_line + "\n")