import os
import av
import argparse
import cv2
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder")
    parser.add_argument("-o", "--output_file")
    args = parser.parse_args()
    # Define the path to the folder containing YOLO format data
    data_folder = args.folder
    # Define the output video file
    output_video = args.output_file

    # Get the list of image files in the data folder
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg') or f.endswith(".png")]

    # Sort the image files to maintain order
    image_files.sort()

    # Create a container for the video
    container = av.open(output_video, 'w')
    w, h = 1920, 1080
    # Define the video codec and parameters
    stream = container.add_stream("h264", 25)
    options = dict(
        threads='0',
        preset='fast',
        profile='high'
    )
    bitrate = 10_000_000
    stream.bit_rate = bitrate
    stream.options = options
    stream.height = int(h)
    stream.width = int(w)

    # Iterate through the image files
    for image_file in tqdm.tqdm(image_files):
        # Read the image
        image_path = os.path.join(data_folder, image_file)
        img = cv2.imread(image_path)

        # Read the corresponding label file
        if 'jpg' in image_file:
            label_path = os.path.join(data_folder, image_file.replace('.jpg', '.txt'))
        elif 'png' in image_file:
            label_path = os.path.join(data_folder, image_file.replace('.png', '.txt'))
        # Open and parse the label file
        with open(label_path, 'r') as label_file:
            lines = label_file.readlines()

        for line in lines:
            # Split the line to get label information
            parts = line.strip().split()

            # YOLO format has (class, x_center, y_center, width, height) for each object
            class_id, x_center, y_center, width, height = map(float, parts)

            # Calculate the coordinates of the bounding box
            x1 = int((x_center - width / 2) * img.shape[1])
            y1 = int((y_center - height / 2) * img.shape[0])
            x2 = int((x_center + width / 2) * img.shape[1])
            y2 = int((y_center + height / 2) * img.shape[0])

            # Draw the bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Class {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the image to RGB format
        img_rgb = cv2.resize(img, (w, h))
        # Create a video frame from the image
        frame = av.VideoFrame.from_ndarray(img_rgb, format='bgr24')

        # Add the frame to the video container
        packet = stream.encode(frame)
        container.mux(packet)

    # Close the video container
    packets = stream.encode()
    container.mux(packets)
    container.close()

    print(f'Video saved as {output_video}')