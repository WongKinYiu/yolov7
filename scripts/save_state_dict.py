import torch

from yolov7.models.experimental import attempt_load
from yolov7.models.yolo import Model


weights = '/media/data/yolov7/yolov7/weights/yolov7-w6.pt'
save_path = '/media/data/yolov7/yolov7/weights/yolov7-w6_state.pt'

model1 = attempt_load(weights, map_location='cpu')
torch.save({'state_dict': model1.state_dict(), 'class_names': model1.names}, save_path)

# test loading model
cfg = '/media/data/yolov7/yolov7/cfg/deploy/yolov7-w6.yaml'
model2 = Model(cfg)
checkpoint = torch.load(save_path, map_location='cpu')
loaded_state_dict = checkpoint['state_dict']
model2.fuse()
model2.load_state_dict(loaded_state_dict)
