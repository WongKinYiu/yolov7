import torch 
import cv2 
import logging
import torch.nn as nn
from utils.general import colorstr

logger = logging.getLogger(__name__)

# unfreezing layers
def unfreezing_layers(named_parameters, layer_id):
    """
    Return unfreezing state of layer

    Args:
        named_parameters :-  model layers module
        layer_id :- layer ID wants to unfreeze.

    return:
        Unfreeze state of layer
    """
    to_be_layer = [f"model.{x}"  for x in layer_id]

    for k, v in named_parameters:
        if any(x in k for x in to_be_layer):
            logger.info(f'Unfreezing {k}')
            v.requires_grad = True

    return  f'Unfreezing Completed for {layer_id}'

def smart_optimizer(model, name = "Adam", lr = 0.001, momentum = 0.9, decay = 1e-5 ):
    pg = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg[2].append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg[0].append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg[1].append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg[0].append(v.im.implicit)
            else:
                for iv in v.im:
                    pg[0].append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg[0].append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg[0].append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg[0].append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg[0].append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg[0].append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg[0].append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg[0].append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg[0].append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg[0].append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg[0].append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg[0].append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg[0].append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg[0].append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg[0].append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg[0].append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg[0].append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg[0].append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg[0].append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg[0].append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg[0].append(v.rbr_dense.vector)
    
    if name == 'Adam':
        optimizer = torch.optim.Adam(pg[0], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(pg[0], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(pg[0], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(pg[0], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')


    optimizer.add_param_group({'params': pg[1], 'weight_decay': decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg[2]})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg[2]), len(pg[1]), len(pg[0])))

    return optimizer

