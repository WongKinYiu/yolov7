"""MIT,
feel free to use and modify,
thanks to the awsome developers that made YoloV7 possible,
made with love by https://github.com/nachovoss"""
import csv
import json
import os
# import time

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression


class ItemCounter:
    """Class for object detection using a YOLOV7 model.
    Export detections as txt, json or csv file.
    Make predictions from image, video and dataset of images.
    All detection functions return a list of dictionaries of detected object classes and counts for each image.
    
    Args:
        img_size (int): Size of the input image (default: 640).
        image_path (str): Path to the input image file (default: 'test_images/1.jpg').
        dataset_path (str): Path to the directory containing the images to be detected (default: None).
        weights_path (str): Path to the pre-trained weights file (default: 'good_models/YOLOv7/best.pt').
        conf_thres (float): Confidence threshold for object detection (default: 0.5).
        iou_thres (float): Intersection over Union (IoU) threshold for non-maximum suppression (default: 0.2).

    Attributes:
        weights_path (str): Path to the pre-trained weights file.
        device (torch.device): Device to run the YOLOv7 model on (CPU or GPU).
        conf_thres (float): Confidence threshold for object detection.
        iou_thres (float): Intersection over Union (IoU) threshold for non-maximum suppression.
        model (torch.nn.Module): YOLOv7 model loaded from the weights file.
        names (list): List of class names for the YOLOv7 model.
        img_size (int): Size of the input image.
        stride (int): Stride of the YOLOv7 model.
        img (numpy.ndarray): Preprocessed image for input to the YOLOv7 model.
        predictions (dict): Dictionary of detected object classes and counts.
        dataset_path (str): Path to the directory containing the images to be detected.

    Methods:
        preprocess_image(): Preprocesses the input image for input to the YOLOv7 model.
        make_predictions(): Runs the YOLOv7 model on the preprocessed image and returns a dictionary of detected object classes and counts.
        make_predictions_from_dataset(): Runs the YOLOv7 model on a dataset of images and returns a list of dictionaries of detected object classes and counts for each image.
        make_predictions_from_video(): Runs the YOLOv7 model on a video and returns a list of dictionaries of detected object classes and counts for each frame.
        export_predictions_to_json(): Exports the predictions to a JSON file.
        export_predictions_to_txt(): Exports the predictions to a txt file.
        export_predictions_to_csv(): Exports the predictions to a CSV file.
    """
    def __init__(self, img_size=640, image_path='PATH TO IMAGE', dataset_path=None, video_path=None, weights_path='PATH TO MODEL', conf_thres=0.5, iou_thres=0.2):
        self.weights_path = weights_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = attempt_load(weights_path, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.model.eval()
        self.img_size = img_size
        self.stride = int(self.model.stride.max())  
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.image_path = image_path
        self.img = self.preprocess_image()
        self.predictions = []
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.function_triggered = False

    def preprocess_image(self, image=None):
        """
        Preprocesses an image for object detection by resizing and transforming it.

        Returns:
            torch.Tensor: The preprocessed image as a tensor.
        """
        if image is None:
            image = cv2.imread(self.image_path)
        else:
            image = image
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        self.img = img
        return self.img

    def make_predictions(self, export_format=None):
        """
        Runs object detection on the image and returns a dictionary of the detected objects and their counts.

        Returns:
            dict: A dictionary containing the detected objects and their counts.
        """
        with torch.no_grad():
            predictions = self.model(self.img, augment=False)[0]
        predictions = non_max_suppression(predictions, self.conf_thres, self.iou_thres, agnostic=False)
        detections_dict = dict()
        for detections in predictions:
            if len(detections):
                for item in detections[:, -1].unique():
                    num_detections = (detections[:, -1] == item).sum()
                    detections_dict[self.names[int(item)]] = num_detections.item()
        if export_format:
            self.predictions = detections_dict
            if export_format == 'txt':
                self.export_predictions_to_file(f'{self.image_path.split("/")[-1]}_predictions.txt')
            if export_format == 'json':
                self.export_predictions_to_json(f'{self.image_path.split("/")[0]}_predictions.json')
            if export_format == 'csv':
                self.export_predictions_to_csv(f'{self.image_path.split("/")[0]}_predictions.csv')
            else:
                raise ValueError(f"Invalid export format: {export_format}. Valid options are 'txt', 'json' and 'csv'.")
        return detections_dict

    def make_predictions_from_dataset(self, export_format=None):
        """
        Runs object detection on a dataset of images and returns a list of dictionaries,
        where each dictionary contains the detected objects and their counts for an image.

        Returns:
            list: A list of dictionaries containing the detected objects and their counts for each image in the dataset.
        """
        pred = []
        for image in os.listdir(self.dataset_path):
            self.image_path = f'{self.dataset_path}/{image}'
            self.preprocess_image()
            detections = self.make_predictions()
            detections['image'] = image
            pred.append(detections)
        self.predictions = pred
        if export_format is not None:
            if export_format == 'txt':
                self.export_predictions_to_file(f'{self.dataset_path.split("/")[0]}_predictions.txt')
            if export_format == 'json':
                self.export_predictions_to_json(f'{self.dataset_path.split("/")[0]}_predictions.json')
            if export_format == 'csv':
                self.export_predictions_to_csv(f'{self.dataset_path.split("/")[0]}_predictions.csv')
            else:
                raise ValueError(f"Invalid export format: {export_format}. Valid options are 'txt', 'json' and 'csv'.")
        return pred

    def make_predictions_from_video(self, frame_interval=30, export_format=None):
        """
        Extracts frames from a video and makes predictions on each frame using the
        object detection model. Returns a list of prediction dictionaries.

        Args:
            frame_interval (int): Interval (in frames) at which predictions should be made.
                Default is 30.
            export_format (str): Format in which to export the predictions. Valid options are
                'txt' and 'json'. If None, no export is done. Default is None.

        Returns:
            list: A list of prediction dictionaries, where each dictionary contains the
            predicted bounding boxes, labels, and scores for a single frame.

        Raises:
            FileNotFoundError: If the video path is invalid.
            ValueError: If the frame interval is not positive.
            IOError: If there was an error reading the video file.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Invalid video path: {self.video_path}")

        if frame_interval <= 0:
            raise ValueError("Frame interval must be positive.")

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            raise IOError(f"Error reading video file: {e}")

        if frame_count < 0:
            frame_count = -frame_count
        frame_count = int(frame_count)

        predictions_list = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if ret and i % frame_interval == 0:
                self.img = self.preprocess_image(frame)
                detections = self.make_predictions()
                detections['image'] = i
                predictions_list.append(detections)
            else:
                continue

        cap.release()
        self.predictions = predictions_list
        if export_format is not None:
            if export_format == 'txt':
                self.export_predictions_to_file(f'{self.video_path.split("/")[-1].split(".")[0]}_predictions.txt')
            elif export_format == 'json':
                self.export_predictions_to_json(f'{self.video_path.split("/")[-1].split(".")[0]}_predictions.json')
            elif export_format == 'csv':
                self.export_predictions_to_csv(f'{self.video_path.split("/")[-1].split(".")[0]}_predictions.csv')
            else:
                raise ValueError(f"Invalid export format{export_format}. Valid options are 'txt', 'json' and 'csv'.")
        return predictions_list

    def export_predictions_to_file(self, file_path):
        """
            Writes the predictions stored in the `self.predictions` attribute to a file
            at the specified `file_path`. If `self.predictions` is a list, each element
            is assumed to represent the predictions for a frame, and the labels and
            counts for each frame are written to the file. If `self.predictions` is a
            dictionary, the labels and counts are written directly to the file.

            Args:
                file_path (str): The path to the file where the predictions will be
                    written.

            Raises:
                TypeError: If `file_path` is not a string.
                IOError: If there is an error writing to the file.

            Returns:
                None.
        """
        with open(file_path, 'w') as f:
            if isinstance(self.predictions, list):
                for frame_predictions in self.predictions:
                    for label, count in frame_predictions.items():
                        f.write(f"{label}: {count}\n")
                    f.write('\n')
            else:
                for label, count in self.predictions.items():
                    f.write(f"{label}: {count}\n")

    def export_predictions_to_json(self, file_path):
        """
            Writes the predictions stored in the `self.predictions` attribute to a JSON
            file at the specified `file_path`. If `self.predictions` is a list, each
            element is assumed to represent the predictions for a frame, and the
            predictions for each frame are written to the file along with the frame's
            image name. If `self.predictions` is a dictionary, the dictionary is
            written directly to the file.

            Args:
                file_path (str): The path to the file where the predictions will be
                    written.

            Raises:
                TypeError: If `file_path` is not a string.
                IOError: If there is an error writing to the file.

            Returns:
                None.
        """
        with open(file_path, 'w') as f:
            if isinstance(self.predictions, list):
                output = []
                for i, frame_predictions in enumerate(self.predictions):
                    item = {}
                    item['image'] = frame_predictions['image']
                    frame_predictions.pop('image')
                    item['predictions'] = frame_predictions
                    output.append(item)
            else:
                output = {"predictions": self.predictions}
            json.dump(output, f, indent=4)

    def export_predictions_to_csv(self, file_path):
        """
        Export the predictions made by the ItemCounter to a CSV file.

        Parameters:
            file_path (str): The path and name of the file to save the CSV data to.

        Returns:
            None
        """
        with open(file_path, mode='w') as csv_file:
            fieldnames = ['label', 'count']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            if isinstance(self.predictions, dict):
                for label, count in self.predictions.items():
                    writer.writerow({'label': label, 'count': count})
            elif isinstance(self.predictions, list):
                for frame_predictions in self.predictions:
                    for label, count in frame_predictions.items():
                        writer.writerow({'label': label, 'count': count})

## Example Usage:
# if __name__ == '__main__':
#     start_time = time.time()
#     item_counter = ItemCounter(video_path="PATH TO VIDEO", dataset_path='PATH TO DATASET', image_path="PATH TO IMAGE", weights_path='PATH TO WEIGHTS', conf_thres=0.5, iou_thres=0.2)
#     predictions = item_counter.make_predictions(export_format='csv')
#     print('predictions: ', predictions)
#     print("--- %s seconds ---" % (time.time() - start_time))
