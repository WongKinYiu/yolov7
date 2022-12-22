from models.experimental import attempt_load
from models.yolo import Model
import torch
import torch.fx
import torch.nn as nn
import numpy as np
import random

from models.common import Conv, DWConv
device = 'cpu'
ckpt = torch.load('/root/workspace/workspace_yolov7/yolov7_training.pt',map_location=device)['model'].state_dict()
model = Model(cfg='/root/workspace/workspace_yolov7/yolov7/cfg/training/yolov7.yaml')
model.load_state_dict(ckpt)
graph = torch.fx.Tracer().trace(model)
traced_model = torch.fx.GraphModule(model, graph)
traced_model_after_npmc = torch.load('/root/workspace/workspace_yolov7/yolov7_training_graphmodule_1221_after_npmc.pt')


# input size is needed to be choosen
input_shape = (1, 3, 640, 640)
random_input = torch.Tensor(np.random.randn(*input_shape))

with torch.no_grad():
    original_output = model(random_input)
    traced_output = traced_model(random_input)
    traced_output_after_npmc = traced_model_after_npmc(random_input)

for o,t in zip(original_output, traced_output_after_npmc):
    assert torch.allclose(o,t), "inference result is not equal!"
print('hi')
# torch.save(traced_model, './yolov7_training_graphmodule.pt')