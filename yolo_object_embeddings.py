import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.ops as ops
import os
class ObjectEmbeddingVisualizer:
    def __init__(self, model, device):
        # # model_type = 'yolov7'
        # # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = torch.hub.load('WongKinYiu/yolov7' if model_type == 'yolov7' else 'ultralytics/yolov5',
        #                           'custom' if model_type == 'yolov7' else 'yolov5s')
        # self.model.to(self.device).eval()
        self.model = model
        self.device = device
    def extract_object_features(self, image, predictions):
        with torch.no_grad():
            # Get feature maps
            if hasattr(self.model, 'model'):
                feature_maps = self.model.model.backbone(image.to(self.device))
            else:
                feature_maps = self.model.backbone(image.to(self.device))
            
            # Get boxes and labels
            boxes = predictions[0].boxes.xyxy  # x1, y1, x2, y2
            labels = predictions[0].boxes.cls
            
            object_features = []
            for scale_idx, feat_map in enumerate(feature_maps):
                # Calculate scale ratio
                scale_h = image.shape[2] / feat_map.shape[2]
                scale_w = image.shape[3] / feat_map.shape[3]
                
                # Scale boxes to feature map size
                scaled_boxes = boxes.clone()
                scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]] / scale_w
                scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]] / scale_h
                
                # ROI pooling
                roi_features = ops.roi_batchool(feat_map, [scaled_boxes.to(self.device)],
                                         output_size=(7, 7))

                # Global average pooling
                pooled_features = F.adaptive_avg_pool2d(roi_features, (1, 1))
                object_features.append(pooled_features.squeeze(-1).squeeze(-1))

            # Concatenate features from all scales
            all_features = torch.cat(object_features, dim=1)
            return all_features.cpu().numpy(), labels.cpu().numpy()

    def extract_object_grounded_features(self, feature_maps, predictions, image_shape: tuple):
        scale = 2
        assert len(image_shape) == 4, 'image shape should be tensor [ch, h, w]'
        embeddings = dict()
        object_cls = list()#dict()
        object_features = []
        try :
            for i_batch, pred in enumerate(predictions):
                # object_features = list()
                for scale_idx, feat_map_all_batches in enumerate(feature_maps): # run over all 3 FM of 3 scales in all batches
                    if scale_idx != scale:
                        continue # take only the last scale
                    feat_map = feat_map_all_batches[i_batch, :, :,:]
                    boxes = pred[:,:4]  # x1, y1, x2, y2
                    labels = pred[:, 5]
                    # Calculate scale ratio
                    scale_h = image_shape[2] / feat_map.shape[1]
                    scale_w = image_shape[3] / feat_map.shape[2]

                    # Scale boxes to feature map size
                    scaled_boxes = boxes.clone()
                    scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]] / scale_w
                    scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]] / scale_h

                    # ROI pooling
                    roi_features = ops.roi_pool(feat_map.float()[None,...], [scaled_boxes.to(self.device)],
                                                output_size=(7, 7))

                    # Global average pooling
                    pooled_features = F.adaptive_avg_pool2d(roi_features, (1, 1))
                    # object_features.append(pooled_features.squeeze(-1).squeeze(-1))
                    [object_features.append(x.squeeze(-1).squeeze(-1)[None,...]) for x in pooled_features]
                    [object_cls.append(x.cpu().numpy()) for x in labels]
                # Concatenate features from all scales
            all_features = torch.cat(object_features, dim=0)
            object_cls = np.array(object_cls)
        except Exception as e:
            raise Exception(f'{i_batch}Error loading data from {i_batch}: {e}\nSee {i_batch}')

            # embeddings.update({i_batch : all_features.cpu().numpy()})
            # object_cls.update({i_batch : labels.cpu().numpy()})

        return all_features, object_cls

    def visualize_object_embeddings(self, features, labels, path, tag=''):
        tsne = TSNE(n_components=2, perplexity=min(30, len(features)-1))
        embeddings_2d = tsne.fit_transform(features.cpu().numpy())
        
        plt.figure(figsize=(10, 10))
        # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
        #                     c=labels, cmap='tab20')
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels)
        plt.colorbar(scatter, label='Object Class')
        plt.title('Object Embeddings labels support {} , classes {}'.format(features.shape[0],
                                                                            np.unique(labels).size))
        plt.show()
        plt.savefig(os.path.join(path,'tsne' + str(tag) + '.png'))

        return embeddings_2d

    def process_image(self, image_tensor):
        predictions = self.model(image_tensor)
        features, labels = self.extract_object_features(image_tensor, predictions)
        embeddings = self.visualize_object_embeddings(features, labels)
        return embeddings, labels
"""
# Usage example
visualizer = ObjectEmbeddingVisualizer()
# Assuming image_tensor is your input image [1, C, H, W]
embeddings, labels = visualizer.process_image(image_tensor)

"""
