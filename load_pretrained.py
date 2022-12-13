import argparse

import numpy as np
import torch


def load_pretrained_weights(yolo4dmodelfile='yolov7-4d-640-nms.pt', yolo5dtrainedmodelfile='yolov7.pt', save_trained_pt='yolov7-4d-trained-640-nms.pt'):
    yolo4d = torch.load(yolo4dmodelfile)
    yolo5d = torch.load(yolo5dtrainedmodelfile)
    model5d = yolo5d['model']
    model4d = yolo4d['model']
    sd5d = model5d.state_dict()
    sd4d = model4d.state_dict()

    sd5d_filtered = {k:v for k,v in sd5d.items() if k in sd4d and sd4d[k].shape == v.shape}
    sd4d.update(sd5d_filtered)
    model4d.load_state_dict(sd4d)

    # checks
    shape_mismatch_parameters = []
    for name, value in yolo4d['model'].state_dict().items():
        assert (name in yolo5d['model'].state_dict()), f"Parameter : {name} not present in pretrained weights"
        o_value = yolo5d['model'].state_dict()[name]
        if value.shape != o_value.shape:
            print(f'Shape mismatch for : {name} pretrained : {o_value.shape} current : {value.shape}')
            shape_mismatch_parameters.append(name)
            assert np.allclose(o_value.cpu().detach().numpy().flatten(), value.cpu().detach().numpy().flatten()), f"Value mismatch for parameter : {name}"
        else:
            assert np.allclose(o_value.cpu().detach().numpy().flatten(), value.cpu().detach().numpy().flatten()), f"Value mismatch for parameter : {name}"


    print(f'All weights loaded from pretrained file except for : {shape_mismatch_parameters} but values are exactly same.')
    torch.save(yolo4d, save_trained_pt)
    print(f'Trained checkpoint .pt file written to : {save_trained_pt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load weights to YOLO model from pretrained weights file")
    parser.add_argument('-m','--model_file', help='Path to untrained yolo model', default='yolov7-4d-640-nms.pt', required=False)
    parser.add_argument('-p','--pretrained_model_file', help='Path to pretrained yolo model', default='yolov7.pt', required=False)
    parser.add_argument('-s','--trained_save_file', help='Path to save the yolo model with trained weights', default='yolov7-4d-trained-640-nms.pt', required=False)
    args = parser.parse_args()

    load_pretrained_weights(args.model_file, args.pretrained_model_file, args.trained_save_file)
