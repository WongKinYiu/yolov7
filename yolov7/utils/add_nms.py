import numpy as np
import onnx
from onnx import shape_inference
try:
    import onnx_graphsurgeon as gs
except Exception as e:
    print('Import onnx_graphsurgeon failure: %s' % e)

import logging

LOGGER = logging.getLogger(__name__)

class RegisterNMS(object):
    def __init__(
        self,
        onnx_model_path: str,
        precision: str = "fp32",
    ):

        self.graph = gs.import_onnx(onnx.load(onnx_model_path))
        assert self.graph
        LOGGER.info("ONNX graph created successfully")
        # Fold constants via ONNX-GS that PyTorch2ONNX may have missed
        self.graph.fold_constants()
        self.precision = precision
        self.batch_size = 1
    def infer(self):
        """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort,
        and fold constant inputs values. When possible, run shape inference on the
        ONNX graph to determine tensor shapes.
        """
        for _ in range(3):
            count_before = len(self.graph.nodes)

            self.graph.cleanup().toposort()
            try:
                for node in self.graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(self.graph)
                model = shape_inference.infer_shapes(model)
                self.graph = gs.import_onnx(model)
            except Exception as e:
                LOGGER.info(f"Shape inference could not be performed at this time:\n{e}")
            try:
                self.graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                LOGGER.error(
                    "This version of ONNX GraphSurgeon does not support folding shapes, "
                    f"please upgrade your onnx_graphsurgeon module. Error:\n{e}"
                )
                raise

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break

    def save(self, output_path):
        """
        Save the ONNX model to the given location.
        Args:
            output_path: Path pointing to the location where to write
                out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        onnx.save(model, output_path)
        LOGGER.info(f"Saved ONNX model to {output_path}")

    def register_nms(
        self,
        *,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        detections_per_img: int = 100,
    ):
        """
        Register the ``EfficientNMS_TRT`` plugin node.
        NMS expects these shapes for its input tensors:
            - box_net: [batch_size, number_boxes, 4]
            - class_net: [batch_size, number_boxes, number_labels]
        Args:
            score_thresh (float): The scalar threshold for score (low scoring boxes are removed).
            nms_thresh (float): The scalar threshold for IOU (new boxes that have high IOU
                overlap with previously selected boxes are removed).
            detections_per_img (int): Number of best detections to keep after NMS.
        """

        self.infer()
        # Find the concat node at the end of the network
        op_inputs = self.graph.outputs
        op = "EfficientNMS_TRT"
        attrs = {
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 0,
        }

        if self.precision == "fp32":
            dtype_output = np.float32
        elif self.precision == "fp16":
            dtype_output = np.float16
        else:
            raise NotImplementedError(f"Currently not supports precision: {self.precision}")

        # NMS Outputs
        output_num_detections = gs.Variable(
            name="num_dets",
            dtype=np.int32,
            shape=[self.batch_size, 1],
        )  # A scalar indicating the number of valid detections per batch image.
        output_boxes = gs.Variable(
            name="det_boxes",
            dtype=dtype_output,
            shape=[self.batch_size, detections_per_img, 4],
        )
        output_scores = gs.Variable(
            name="det_scores",
            dtype=dtype_output,
            shape=[self.batch_size, detections_per_img],
        )
        output_labels = gs.Variable(
            name="det_classes",
            dtype=np.int32,
            shape=[self.batch_size, detections_per_img],
        )

        op_outputs = [output_num_detections, output_boxes, output_scores, output_labels]

        # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
        # become the final outputs of the graph.
        self.graph.layer(op=op, name="batched_nms", inputs=op_inputs, outputs=op_outputs, attrs=attrs)
        LOGGER.info(f"Created NMS plugin '{op}' with attributes: {attrs}")

        self.graph.outputs = op_outputs

        self.infer()

    def save(self, output_path):
        """
        Save the ONNX model to the given location.
        Args:
            output_path: Path pointing to the location where to write
                out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        onnx.save(model, output_path)
        LOGGER.info(f"Saved ONNX model to {output_path}")
