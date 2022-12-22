from models.experimental import attempt_load
from models.yolo import Model
import torch
import torch.fx
import torch.nn as nn
import numpy as np
import random

from models.common import Conv, DWConv

###
import torch.nn as nn
class detectConfig(nn.Module):
    def __init__(self,na,nc,nl,anchors,stride,anchor_grid, grid, no):
        super(detectConfig, self).__init__()
        self.na = na
        self.nc = nc
        self.nl = nl
        self.anchors = anchors
        self.stride = stride
        self.anchor_grid = anchor_grid
        self.grid = grid
        self.no = no
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    
    def post_processing(self, x):
        z = []
        for i in range(self.nl):
            print(x[i].shape)
            bs, _, ny, nx, _  = x[i].shape
            y = x[i]
            # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
            #     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            # y = x[i].sigmoid()
            # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            z.append(y.view(bs, -1, self.no))      
        return z
        return (torch.cat(z,1), x)
# save configurations
model = Model(cfg='/root/workspace/workspace_yolov7/yolov7/cfg/training/yolov7.yaml')
stride = model.stride
names = model.names
idetect = model.model[-1]
dconfig = detectConfig(na=idetect.na, nc=idetect.nc, nl=idetect.nl, anchors=idetect.anchors, stride=idetect.stride,
                            anchor_grid=idetect.anchor_grid, grid=idetect.grid, no=idetect.no)
# load graphmodule model
model = torch.load('/root/workspace/workspace_yolov7/yolov7_training_graphmodule_1221_after_npmc.pt')
# attach configurations
model.stride = stride
model.names = names
model.model = nn.Sequential(dconfig)

fused_model = torch.load('/root/workspace/workspace_yolov7/yolov7_training_graphmodule_1221_after_npmc_fused.pt')
fused_model.stride = stride
fused_model.names = names
fused_model.model = nn.Sequential(dconfig)
###
ckpt = torch.load('/root/workspace/workspace_yolov7/yolov7_training.pt',map_location='cpu')['model'].state_dict()
original = Model(cfg='/root/workspace/workspace_yolov7/yolov7/cfg/training/yolov7.yaml')
original.load_state_dict(ckpt)


# input size is needed to be choosen
input_shape = (1, 3, 640, 640)
random_input = torch.Tensor(np.random.randn(*input_shape))

### Train
model.train()
original.train()

with torch.no_grad():
    original_output = original(random_input)
    traced_output = model(random_input)
for o,t in zip(original_output, traced_output):
    print(torch.allclose(o,t))
    assert torch.allclose(o,t), "inference result is not equal!"

### Evaluation
original.eval()
model.eval()
with torch.no_grad():
    original_output = original(random_input)
    traced_output = model(random_input)
    traced_output_after_pp = dconfig.post_processing(traced_output)

for o,t in zip(original_output[0], traced_output_after_pp[0]):
    print(torch.allclose(o,t))
    assert torch.allclose(o,t), "inference result is not equal!"

# torch.save(traced_model, './yolov7_training_graphmodule.pt')