import torch
from .experimental import End2End, ONNX_ORT, ONNX_TRT
import torch.nn as nn
from .yolo import Model
import torchvision
from utils.general import non_max_suppression
# from utils.general_rv import non_max_suppression # HK@@ overide Yuval
from PIL import Image

@torch.jit.script
def min_value_100_helper(x):
    min_val = torch.min(torch.tensor([100, x.size(0)], device=x[0].device))
    return min_val


class End2EndRVFixedOutput(nn.Module):
    def __init__(self, model: torch.nn.Module, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__.update(model.__dict__)
        device = device if device else torch.device('cpu')
        self.model = model.to(device)

    def forward(self, x,  **kwargs):
        x = self.model(x, **kwargs)  # forward in End2End model
        # expand tensor dimensions
        # Expand the first dimension to 100
        expanded_size = (100,) + x.size()[1:]
        expanded_tensor = torch.zeros(expanded_size, device=x.device)

        # Copy the values from the original tensor to the expanded tensor
        # num_to_copy = min(100, x.size(0))
        num_to_copy = min_value_100_helper(x)

        expanded_tensor[:num_to_copy] = x[:num_to_copy]
        # index_num_to_copy = torch.tensor(num_to_copy, device=x.device)
        # expanded_tensor.index_copy_(0, torch.arange(num_to_copy, device=x.device, dtype=torch.int32), x[:num_to_copy])
        x = expanded_tensor
        # x = expanded_tensor.unsqueeze(0)  # HK@@ tensorleap extension

        return x


# @torch.jit.script
def forward_nms(x):
    # conf_thres = 0.5
    # iou_thres = 0.5
    x = non_max_suppression(x)  # list of detections, on (n,6) tensor per image [xyxy, conf, cls], conf_thres, iou_thres
    return x


class End2EndRVFixedOutput_TorchScript(nn.Module):
    def __init__(self, model: torch.nn.Module, device=None, *args, **kwargs):
        # input model without NMS (like ModelRV)
        super().__init__(*args, **kwargs)
        # self.__dict__.update(model.__dict__)
        device = device if device else torch.device('cpu')
        self.model = model.to(device)
        self.model.model[-1].end2end = True

    def forward_yolo_no_nms(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        # x = self.model(x)
        x = self.forward_yolo_no_nms(x)
        x = forward_nms(x=x)
        return x


class End2EndRVFloatOutput_TRT(nn.Module):
    def __init__(self, model: torch.nn.Module, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__.update(model.__dict__)
        device = device if device else torch.device('cpu')
        self.model = model.to(device)

    def forward(self, x,  **kwargs):
        x = self.model(x, **kwargs)
        x = (torch.tensor(x[0], dtype=torch.float32), x[1], x[2], torch.tensor(x[3], dtype=torch.float32))
        return x


@torch.jit.script
def fill_helper(x, num_det):
    for n in range(num_det.shape[0]):
        x[n, int(num_det[n]):] = torch.tensor(-1.).to(x.device)
    return x


class End2EndRVFillOutput_TRT(nn.Module):
    def __init__(self, model: torch.nn.Module, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__.update(model.__dict__)
        device = device if device else torch.device('cpu')
        self.model = model.to(device)

    def forward(self, x,  **kwargs):
        x = self.model(x, **kwargs)
        # x_cls_fill = x[3]
        # for n in range(x[0].shape[0]):
        #     x[3][n, x[0][n]:] = -1
        # x[3] = fill_helper(x[3], x[0])
        x = (torch.tensor(x[0], dtype=torch.float32), x[1], x[2], torch.tensor(fill_helper(x[3], x[0]), dtype=torch.float32))
        return x


class End2EndRVFixedOutput_TRT(nn.Module):
    def __init__(self, model: torch.nn.Module, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__.update(model.__dict__)
        device = device if device else torch.device('cpu')
        self.model = model.to(device)

    def forward(self, x,  **kwargs):
        x = self.model(x, **kwargs)  # forward in End2End model
        # cat to [Image number, x0, y0, x1, y1, class id, confidence]
        valid_detections = torch.zeros_like(x[3], device=x[0].device)
        # valid_detections[:, :x[0]] = 1
        for i in range(x[0].shape[0]):
            valid_detections[i, :x[0][i]] = i  # 1
        num_detections = x[0]
        # x = torch.cat( (img_num, x[1][0], x[3][0], x[2][0]), dim=0)
        x = torch.cat((valid_detections.unsqueeze(2), x[1], x[3].unsqueeze(2), x[2].unsqueeze(2)), dim=2)
        # expand tensor dimensions
        # Expand the first dimension to 100
        # x = x.squeeze()
        # expanded_size = (100,) + x.size()[1:]
        # expanded_size =  x.size()
        # expanded_tensor = torch.zeros(expanded_size, device=x.device)
        expanded_tensor = torch.zeros((1, 100, 7), device=x.device)
        expanded_tensor = expanded_tensor.squeeze()
        # Copy the values from the original tensor to the expanded tensor
        # num_to_copy = min(100, x.size(0))
        # num_to_copy = min(100, num_detections[0])
        # num_to_copy = torch.min(torch.tensor(100, device=x[0].device), num_detections)
        num_to_copy = num_detections # torch.min(torch.tensor(x.shape[1], device=x[0].device), num_detections)
        # expanded_tensor[:num_to_copy[0]]= x[:num_to_copy[0]]
        # expanded_tensor[:num_to_copy[0]]= x[0, :num_to_copy[0]]
        for n in range(num_to_copy.shape[0]):
            if n == 0:
                expanded_tensor[:num_to_copy[n]]= x[n, :num_to_copy[n]]
            else:
                expanded_tensor[num_to_copy[n-1]:(num_to_copy[n]+num_to_copy[n-1])] = x[n, :num_to_copy[n]]
        # index_num_to_copy = torch.tensor(num_to_copy, device=x.device)
        # expanded_tensor.index_copy_(0, torch.arange(num_to_copy, device=x.device, dtype=torch.int32), x[:num_to_copy])

        x = expanded_tensor

        return x


class End2EndRV2ModelsCombineFixedOutput(nn.Module):
    def __init__(self, model1: torch.nn.Module, model2: torch.nn.Module, device=None, *args, **kwargs):  #
        super().__init__()
        # self.__dict__.update(model1.__dict__)
        self.names = ['person', 'car', 'train', 'truck', 'bike', 'animal', 'locomotive', 'braking_shoe']
        device = device if device else torch.device('cpu')
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)

    def forward(self, x,  **kwargs):
        # x2 = super().forward(x, **kwargs)  # forward in End2End model
        x1 = self.model1(x)
        x2 = self.model2(x)

        # fix model 2 classes:
        x2[:, 5] += 4  # 6

        x = torch.cat((x1, x2), dim=0)

        # Concatenate the tensors from the two tuples
        # resulting_tensor = torch.cat((x1[0], x2[0]), dim=1)

        # Concatenate the tensors within the lists
        # resulting_tensor_list = [torch.cat((t1, t2), dim=1) for t1, t2 in zip(x1[1], x2[1])]

        # x = (resulting_tensor, resulting_tensor_list)
        # x= x1

        # expand tensor dimensions
        # Expand the first dimension to 100
        expanded_size = (100,) + x.size()[1:]
        expanded_tensor = torch.zeros(expanded_size, device=x.device)

        # Copy the values from the original tensor to the expanded tensor
        # num_to_copy = min(100, x.size(0))
        # num_to_copy = torch.min(torch.tensor([100, x.size(0)], device=x[0].device))
        num_to_copy = min_value_100_helper(x)

        expanded_tensor[:num_to_copy] = x[:num_to_copy]

        x = expanded_tensor

        return x


class End2EndRVTwoModels(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model1, model2, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None, n_classes=80):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        # First model
        self.model1 = model1.to(device)
        self.model1.model[-1].end2end = True
        # Second model
        self.model2 = model2.to(device)
        self.model2.model[-1].end2end = True
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device, n_classes)
        self.end2end.eval()

    def forward(self, x):
        # forward in models
        x1 = self.model1(x)
        x2 = self.model2(x)

        # Expand model 1 with more slots for the size of the classes in model 2
        num_labels_model2 = len(self.model2.names)
        x1_ext = torch.zeros(x1.shape[0], x1.shape[1], num_labels_model2, device=x1.device)
        x1 = torch.cat((x1, x1_ext), dim=2)

        # Expand model 2 with more slots for the size of the classes in model 1
        num_labels_model1 = len(self.model1.names)
        x2_ext = torch.zeros(x2.shape[0], x2.shape[1], num_labels_model1, device=x2.device)
        x2temp = torch.cat((x2[:, :, 0:5], x2_ext), dim=2)
        x2 = torch.cat((x2temp, x2[:, :, 5:]), dim=2)

        # cat to single tensor
        x = torch.cat((x1, x2), dim=1)

        # end2end nms:
        x = self.end2end(x)

        return x

class Model_inp_wrapper_2nd(Model):
    def __init__(self, model:torch.nn.Module, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.__dict__.update(model.__dict__)

    def forward(self, x, switch_ch_dim: bool = True, **kwargs):
        #if image from cv2.imread (onnx)
        if switch_ch_dim:
            x = torch.permute(x, (0, 3, 1, 2))

        x = super().forward(x, **kwargs)

        return x

#
# class End2EndRVForwardSwitch(nn.Module):
#     '''export onnx for forward switch from localization Yolo model (localization_model) and resnet classification model for state (state_model).'''
#     def __init__(self, localization_model: torch.nn.Module, device=None, *args, **kwargs):  # state_model: torch.nn.Module,
#         super().__init__()
#         # self.__dict__.update(model1.__dict__)
#         self.names = ['switch_left', 'switch_right', 'switch_unknown']
#         device = device if device else torch.device('cpu')
#         self.localization_model = localization_model.to(device)
#         # self.state_model = state_model.to(device)
#
#     def forward(self, x):
#         input_img = x
#         # forward in yolo end2end model (with nms):
#         detections = self.localization_model(x)  # detections: [batch number, x0, y0, x1, y1, class id, confidence]
#         num_detections = detections.size(0)
#         # loop over detections, crop image by detection bbox for classification state model
#         for d in range(num_detections):
#             bbox = detections[d, 1:5]
#             cropped_region = input_img[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
#             # cropped_region = input_img
#             # DEBUG check crop:
#             # Convert the tensor to a PIL Image
#             cropped_region_cpu = cropped_region.cpu()
#             cropped_region_cpu = cropped_region_cpu.squeeze(0)
#             image_pil = Image.fromarray(
#                 (cropped_region_cpu * 255).byte().numpy())  # Assuming the tensor values are in the range [0, 1]
#             # Save the PIL Image to a file
#             image_pil.save('image_crop.jpg')
