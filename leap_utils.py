from matplotlib import patches
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, List, Union, Dict
import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy._typing import NDArray
from leapcfg.config import CONFIG, hyp, data_dict


def xyxy_to_xywh_format(boxes: Union[NDArray[np.float32], tf.Tensor]) -> Union[NDArray[np.float32], tf.Tensor]:
    """
    This gets bb in a [X,Y,W,H] format and transforms them into an [Xmin, Ymin, Xmax, Ymax] format
    :param boxes: [Num_boxes, 4] of type ndarray or tensor
    :return:
    """
    min_xy = (boxes[..., :2] + boxes[..., 2:]) / 2
    max_xy = (boxes[..., 2:] - boxes[..., :2])
    if isinstance(boxes, tf.Tensor):
        result = tf.concat([min_xy, max_xy], -1)
    else:
        result = np.concatenate([min_xy, max_xy], -1)
    return result


def draw_image_with_boxes(image, bounding_boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        confidence, label = bbox.confidence, bbox.label

        # Convert relative coordinates to absolute coordinates
        abs_x = x * image.shape[1]
        abs_y = y * image.shape[0]
        abs_width = width * image.shape[1]
        abs_height = height * image.shape[0]

        # Create a rectangle patch
        rect = patches.Rectangle(
            (abs_x - abs_width / 2, abs_y - abs_height / 2),
            abs_width, abs_height,
            linewidth=2, edgecolor='r', facecolor='none'
        )

        # Add the rectangle to the axes
        ax.add_patch(rect)

        # Display label and confidence
        ax.text(abs_x - abs_width / 2, abs_y - abs_height / 2 - 5,
                f"{label} {confidence:.2f}", color='r', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Show the image with bounding boxes
    plt.show()


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    labels = data_dict['names']
    if not isinstance(bb_array, np.ndarray):
        bb_array = np.array(bb_array)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][5] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][:4])
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][4]
            class_idx = bb_array[i][5]
            if class_idx > len(labels) - 1:
                print("Class idx is larger than the number of supplied labels")
            label_name = labels[int(class_idx)]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=str(label_name))

            bb_list.append(curr_bb)
    return bb_list
