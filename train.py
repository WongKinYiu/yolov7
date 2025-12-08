import argparse
import logging
import math
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

import test  # import test.py to get mAP after each epoch

try:
#    from yolov7_main.models.common import Conv, DWConv
 #   from yolov7_main.utils.google_utils import attempt_download
    from models.experimental import attempt_load

except:
    print("", 100 * '==')
    print(os.getcwd())
    import sys
    sys.path.append('/home/hanoch/projects/tir_od')
    from tir_od.yolov7.models.experimental import attempt_load
#

from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, reset_dataloader_batch_size
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_img_size, \
    print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_results, plot_evolution, append_to_txt
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from clearml import Task, Logger
logger = logging.getLogger(__name__)
clear_ml = True


if clear_ml:  # clearml support

    task = Task.init(
            project_name="TIR_OD",
            task_name="train yolov7 'locomotive' class"  # output_uri = True model torch.save will uploaded to file server or =/mnt/myfolder or AWS or Azure
            # output_uri='azure://company.blob.core.windows.net/folder'
    )
    # Task.execute_remotely() will invoke the job immidiately over the remote and not DeV
    task.set_base_docker(docker_image="nvcr.io/nvidia/pytorch:24.09-py3", docker_arguments="--shm-size 8G")
#     clear_ml can capture graph like tensorboard



gradient_clip_value = 100.0
opt_gradient_clipping = True


class OhemScheduler():
    def __init__(self, ohem_periodicity, ohem_start_ep, total_epochs):
        self.ohem_on_epochs = range(ohem_start_ep, total_epochs, ohem_periodicity)
        self.ohem_is_active = False

    def is_ohem_active(self, curr_epoch):

        if curr_epoch in self.ohem_on_epochs: # toggle
            self.ohem_is_active = not(self.ohem_is_active)
        return self.ohem_is_active

    
def callback_fun_det_anomaly():
    pass
def find_clipped_gradient_within_layer(model, gradient_clip_value):
    margin_from_sum_abs = 1 / 3
    # find if excess gradient value w/o clipping using the clipping API with clip=INF=100 :just check total norm with dummy high clip val
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    if total_grad_norm > gradient_clip_value:
        max_grad_temp = -100.0
        name_grad_temp = 'None'

        for name, param in model.named_parameters():
            # not_none_grad = [p is not None for p in param.grad]
            if param.grad is not None:
                # print(param.grad)
                norm_layer = torch.unsqueeze(torch.norm(param.grad.detach(), float(2)), 0)
                not_none_grad = [i for i in norm_layer if i is not None]
                for u in not_none_grad:
                    if (u>gradient_clip_value*margin_from_sum_abs).any():
                        # print(name, u[u > gradient_clip_value/2])
                        if (u[u > gradient_clip_value*margin_from_sum_abs] > max_grad_temp):
                            max_grad_temp = u[u > gradient_clip_value *margin_from_sum_abs]
                            name_grad_temp = name

        print("layer {} with max gradient {}".format(name_grad_temp, max_grad_temp))

def compare_models_basic(model1, model2):
    for ix, (p1, p2) in enumerate(zip(model1.parameters(), model2.parameters())):
        if p1.data.ne(p2.data).sum() > 0:
            print('Models are different', ix, p1.data.ne(p2.data).sum())
            return False
    return True


def compare_models(model1, model2):
    # Iterate through named layers and parameters of both models
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Layer names differ: {name1} vs {name2}")


        # Compare the parameters
        if not torch.equal(param1, param2):
            print('Difference found in layer{}  {}'.format(name1, param1.data.ne(param2.data).sum()))

    return
    # print("No differences found in any layer.")


def train(hyp, opt, device, tb_writer=None, clml_logger=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    is_torch_240 = int(re.search(r'([\d.]+)', torch.__version__).group(1).replace('.', '')) >=240
    model_name = str(opt.weights)[str(opt.weights).find('yolo'):].split('/')[0]

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    if opt.predefined_seed:
        hyp['seed'] = 2 + rank
        init_seeds(2 + rank)
    else:
        rand_seed = int(time.time())
        hyp['seed'] = rand_seed
        init_seeds(rand_seed)

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)

    if clear_ml: #clearml support
        config_file = task.connect_configuration(opt.data)
        with open(config_file) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        # data_dict = task.connect_configuration(data_dict)
    else:
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    with open(save_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_dict, f, sort_keys=False)

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=opt.input_channels, nc=nc, anchors=hyp.get('anchors')).to(device)  # create model structure according to yaml and not the checkpoint
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=opt.input_channels, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    images_parent_folder = data_dict['path']
    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = opt.nom_batch_size_grad_acm #64  # nominal batch size # HK gradient accumulation fixed size no less than
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases # also need to be set to zero
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam: # @@ HK AdamW() is a fix for Adam due to Wdecay/L2 loss bug
        optimizer = optim.AdamW(pg0, lr=hyp['lr0'], weight_decay=0 , betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], weight_decay=0 , momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay : only over weights
    optimizer.add_param_group({'params': pg2 , 'weight_decay': 0})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))

    # validate that we considered every parameter
    # param_dict = {pn: p for pn, p in model.named_parameters()}
    # inter_params = set(pg1) & set(pg0) & set(pg1)
    # union_params = set(pg1) | set(pg0) | set(pg1)
    # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    # assert len(
    #     param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
    #                                             % (str(param_dict.keys() - union_params),)


    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # To modify the LambdaLR scheduler so that the learning rate increases when accuracy drops, you can introduce a trigger-based mechanism. Below is the modified implementation that:
    # def lr_lambda(epoch):
    #     increase_factor = 2
    #     base_lr = (1 - epoch / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    #     if accuracy_drop_trigger:
    #         return base_lr * increase_factor  # Increase LR
    #     return base_lr  # Normal decay

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)


    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)  => HK@@ stride for letterbox reshape to multiple of 32 in YOLO
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                            rel_path_images=images_parent_folder, num_cls=data_dict['nc'])
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)


    with open(save_dir / 'trainig_set.txt', 'w') as f:
        for file in dataset.img_files:
            f.write(f"{file}\n")

    # Process 0
    if rank in [-1, 0]:
        testloader , test_dataset = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=False, rank=-1, # @@@ rect was True why?
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '),
                                       rel_path_images=images_parent_folder, num_cls=data_dict['nc'])

        mlc = np.concatenate(test_dataset.labels, 0)[:, 0].max()  # max label class
        assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (        mlc, nc, opt.data, nc - 1)

        with open(save_dir / 'test_set.txt', 'w') as f:
            for file in test_dataset.img_files:
                f.write(f"{file}\n")

        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes

        labels_test = np.concatenate(testloader.dataset.labels, 0)
        c_test = torch.tensor(labels_test[:, 0])  # classes

        if not opt.resume:
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            if opt.amp or 1:
                model.half().float()  # pre-reduce anchor precision TODO HK Why ? >???!!!!
    if 1:
        print("opt.local_rank", opt.local_rank)
        print(100*'++')
    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    class_inverse_freq = labels_to_class_weights(dataset.labels, nc).to(device)
    model.names = names
    maps_val_all = list()
    stop_train_plot_image = False
    enable_cont_saving_images = False
    # Start training
    t0 = time.time()
    if hyp['warmup_epochs'] !=0: # otherwise it is forced to 1000 iterations
        nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations) # HK@@ bad for overfitting test where few examples i.e itoo few iterations
    else:
        nw = 0


    if opt.cosine_anneal: # override the Lr schem
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                    # T_0 period of 1st wamup Number of iterations for the first restart  ;T_mult=1 increase T_0 each period
                                                                    T_0=int(2*hyp['warmup_epochs']), T_mult=2,
                                                                    eta_min=0, #hyp['lr0'] / 10,
                                                                    last_epoch=-1)  # lr range test take max warmup/4 for CLR https://arxiv.org/abs/1803.09820s

    if 0:
        from utils.plots import plot_lr_scheduler
        plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=save_dir)

    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    if 1:
        scaler = amp.GradScaler(enabled=cuda)
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=opt.amp) if is_torch_240 else torch.cuda.amp.GradScaler(enabled=opt.amp)

    loss_weight = torch.tensor([]) # for BCE
    if opt.multi_class_no_multi_label:
        loss_weight = torch.ones(1)

    if opt.loss_weight:
        loss_weight = class_inverse_freq
        if 0:
            loss_weight = torch.Tensor([0.0, 0.0, 1.0]).to(device)
    #     Replaced YOLO classification loss with Focal Loss using per-class α values. Kept Objectness Loss and BBox Loss unchanged.
    if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
        compute_loss_ota = ComputeLossOTA(model, loss_weight=loss_weight)  # init loss class
        if opt.multi_class_no_multi_label:
            raise ValueError('Not imp yet!')

    compute_loss = ComputeLoss(device, model, loss_weight=loss_weight,
                               multi_class_no_multi_label=opt.multi_class_no_multi_label,
                               multi_label_asymetric_focal_loss=opt.multi_label_asymetric_focal_loss)  # init loss class it is required for the test set as well hance mandatory

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    if (not opt.nosave):
        torch.save(model, wdir / 'init.pt')
    # from pympler import tracker
    # the_tracker = tracker.SummaryTracker()
    # the_tracker.print_diff()
    # OP
    # the_tracker.print_diff()

    if 0: # HK TODO remove later  The anomaly mode tells you about the nan. If you remove this and you have the nan error again, you should have an additional stack trace that tells you about the forward function (make sure to enable the anomaly mode before the you run the forward).
        torch.autograd.set_detect_anomaly(True)

    print(100 * '==')
    print('Training set labels {} count : {}'.format(names, torch.bincount(c.long(), minlength=nc) + 1))

    print(100 * '==')
    print('Validation set labels {} count : {}'.format(names, torch.bincount(c_test.long(), minlength=nc) + 1))

    if opt.ohem_start_ep > 0:
        ohem_scheduler = OhemScheduler(ohem_periodicity=opt.ohem_period,
                                        ohem_start_ep=opt.ohem_start_ep, total_epochs=epochs)
        # Run inference over the new model
        dataloader_ohem_eval_set = copy.deepcopy(dataloader)
        ep_before_ohem = 0

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        try:
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", e)

        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # print(np.unique(targets, return_counts=True))
            # print(np.bincount(targets[:,1].long(), minlength=nc))
            ni = i + nb * epoch  # number integrated batches (since train start) i.e. iterations

            imgs = imgs.to(device, non_blocking=True).float()

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # with amp.autocast(enabled=cuda): # to decrease GPU VRAM turn off OTA loss see what happen HT TODO ::
            # with amp.autocast(enabled=(cuda and opt.amp)):
            with amp.autocast(enabled=(cuda)):
                pred = model(imgs)  # forward [B, C, W,H, [bbox[4], objectness[1], class-conf[nc]]]
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # HK TODO : https://discuss.pytorch.org/t/switching-between-mixed-precision-training-and-full-precision-training-after-training-is-started/132366/4    remove scaler backwards
            # Backward
            scaler.scale(loss).backward()
            # gradient clipping find and clip
            if opt_gradient_clipping:
                if 1: # args.ams
                    # find_clipped_gradient_within_layer(model, gradient_clip_value)
                    if ni > nw and rank in [-1, 0]:
                        if ni % accumulate == 0: # same condition as for the scaler.update() to synch
                            scaler.unscale_(optimizer)
                            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                             gradient_clip_value)  # dont worry the clipping occurs if |sum(grad)|^2>1000 => no clipping just monitoring
                            tb_writer.add_scalar('Grad norm', total_grad_norm, ni)
                            # if total_grad_norm > gradient_clip_value:
                            #     print("Gradeint {} was clipped to {}".format(total_grad_norm, gradient_clip_value))
                else:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                     gradient_clip_value)  # dont worry the clipping occurs if |sum(grad)|^2>1000 => no clipping just monitoring

                    tb_writer.add_scalar('Grad norm', total_grad_norm, ni)

            last_lr = scheduler.get_last_lr()
            # print(last_lr)
            tb_writer.add_scalar('last_lr', last_lr[-1], ni)

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # import tifffile
                # for ix, img in enumerate(imgs):
                #     print(ix, torch.std(img), torch.quantile(img, 0.5))
                #     tifffile.imwrite(os.path.join('/home/hanoch/projects/tir_od/outputs', 'img_scl_bef_mosaic' + str(ix)+'.tiff'),
                #                      img.cpu().numpy().transpose(1, 2, 0))
                #

                # Plot
                if ((plots and ni < 100) or enable_cont_saving_images) and not(stop_train_plot_image):
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f, None, 640, 16, clml_logger), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        if epoch >= opt.ohem_start_ep and opt.ohem_start_ep >0:

            # Run inference over the new model
            # dataloader_orig = copy.deepcopy(dataloader)
            if ohem_scheduler.is_ohem_active(epoch):
                ohem_batch_size = 1 # since reduction inside loss comp is per 1-3 scales of the classification head hence B axis is diminished
                # orig_dataloader_batch_size = dataloader.batch_size
                # Restore dataloader to its basic
                dataloader = copy.deepcopy(dataloader_ohem_eval_set)
                # modified version for eval OHEM top-k
                dataloader_ohem_eval_set = reset_dataloader_batch_size(dataloader, ohem_batch_size, disable_augment=True)


                results, maps, times, loss_per_image_acm = test.test(data_dict,
                                                 batch_size=ohem_batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 save_json=opt.save_json,
                                                 model=ema.ema,
                                                 iou_thres=hyp['iou_t'],
                                                 single_cls=opt.single_cls,
                                                 dataloader=dataloader_ohem_eval_set,
                                                 save_dir=save_dir,
                                                 verbose=False,
                                                 plots=False,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric,
                                                 hyp=hyp,
                                                 model_name=model_name)

                # restore original dataloader from now on the dataloader will be modified inplace to accomodate OHEM top-k indices
                dataloader_ohem_eval_set = copy.deepcopy(dataloader)

                # dataloader.batch_size = orig_dataloader_batch_size # restore
                loss_per_image_acm = torch.stack(loss_per_image_acm)
                val, top_k_indices = torch.topk(loss_per_image_acm.T, int(opt.ohem_topk * dataloader.dataset.__len__()))
                dataloader.dataset.resample_ohem(top_k_indices=top_k_indices)

                print('OHEM epoch{} top-k {}'.format(epoch, val))
            else:
                print('OHEM OFF epoch{}'.format(epoch))

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        # print("Lr : ", 10*'+',lr)
        scheduler.step()
        if 1:  #@@ HK
            plots = True
        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs


            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times, _ = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 save_json=opt.save_json,
                                                 model=ema.ema,
                                                 iou_thres=hyp['iou_t'],
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric,
                                                 hyp=hyp,
                                                 model_name=model_name)


            if epoch > 1:
                if maps_val_all[-1][2]-maps[2] >= 0.08: # abrupt falling of 3rs class locomotive
                    stop_train_plot_image = True
                    print('!!!!!!!!!!!!!!!!!!!!!  abrupt failure in map  !!!!!!!!!!!!!!!')
            maps_val_all.append(maps)
            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B
            for i, val in enumerate(maps):
                tb_writer.add_scalar(f'map_50% class {names[i]}', val, ni)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1), w=[0.0, 0.0, 1.0, 0.0])  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
                file_path = os.path.join(save_dir,
                                         'best_fitness.txt')
                formatted_line = f"{best_fitness.item():3e}\n"
                # Open the file in append mode and write the new line
                with open(file_path, 'a') as file:
                    file.write(formatted_line)
                # 'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'

            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),  # HK TODO hlaf() is only if AMP is True
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimerizs
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--save-json', action='store_true', help=' save save-json')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify') #Changed in version 2.0.0: The launcher will passes the --local-rank=<rank> argument to your script. From PyTorch 2.0.0 onwards, the dashed --local-rank is preferred over the previously used underscored --local_rank.
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save-period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--norm-type', type=str, default='standardization',
                                        choices=['standardization', 'single_image_0_to_1', 'single_image_mean_std','single_image_percentile_0_255',
                                                 'single_image_percentile_0_1', 'remove+global_outlier_0_1'],
                                        help='Normalization approach')
    parser.add_argument('--no-tir-signal', action='store_true', help='')

    parser.add_argument('--tir-channel-expansion', action='store_true', help='drc_per_ch_percentile')

    parser.add_argument('--input-channels', type=int, default=3, help='')

    parser.add_argument('--save-path', default='/mnt/Data/hanoch', help='save to project/name')

    parser.add_argument('--gamma-aug-prob', type=float, default=0.1, help='')

    parser.add_argument('--fl-gamma', type=float, default=-1, help='')

    parser.add_argument('--amp', action='store_true', help='Remove torch AMP')

    parser.add_argument('--predefined-seed', action='store_true', help='predefined_seed only set it to constant otherwise add args that load the random one ')

    parser.add_argument('--csv-metadata-path', default='', help='save to project/name')

    parser.add_argument('--loss-weight', action='store_true', help='weight the loss by 1/freq to compensate for imbalanced data')

    parser.add_argument('--embed-analyse', action='store_true', help='')

    parser.add_argument('--ohem-start-ep', type=int, default=-1, help='Online Hard example by re-eval training set after N-iters and take top-K')

    parser.add_argument('--ohem-topk', type=float, default=0.7, help='')

    parser.add_argument('--ohem-period', type=int, default=5, help='Online Hard example N epoces toggeling on/off duration')

    parser.add_argument('--nom-batch-size-grad-acm', type=int, default=64, help='')

    parser.add_argument('--cosine-anneal', action='store_true', help='')

    parser.add_argument('--multi-class-no-multi-label', action='store_true', help='disbale multi-label')

    parser.add_argument('--multi-label-asymetric-focal-loss', action='store_true', help='disbale multi-label')

    opt = parser.parse_args()
    # Only for clearML env

    clml_logger = Logger.current_logger()

    if opt.multi_class_no_multi_label and opt.multi_label_asymetric_focal_loss:
        raise ValueError('ASL is for multi label rather than multi class')

    if opt.tir_channel_expansion: # operates over 3 channels
        opt.input_channels = 3

    if opt.tir_channel_expansion and opt.norm_type != 'single_image_percentile_0_1': # operates over 3 channels
        print('Not a good combination')

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        if opt.save_path == '':
            opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
        else:
            opt.save_dir = increment_path(os.path.join(opt.save_path, Path(opt.project) , opt.name), exist_ok=opt.exist_ok | opt.evolve)

            # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    defualt_random_pad = True # lazy hyp def

    # clearml support
    if clear_ml: #clearml support
        config_file = task.connect_configuration(opt.hyp, name='hyperparameters_cfg')
        with open(config_file) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        print("", 100 * '==')
        print('Hyperparameters:', hyp)
    else:
        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    #defaults for backward compatible hyp files whree not set
    hyp['person_size_small_medium_th'] = hyp.get('person_size_small_medium_th', 32 * 32)
    hyp['car_size_small_medium_th'] = hyp.get('car_size_small_medium_th', 44 * 44)
    hyp['random_pad'] = hyp.get('random_pad', defualt_random_pad)
    if opt.fl_gamma > 0:
        hyp['fl_gamma'] = opt.fl_gamma
    else:
        hyp['fl_gamma'] = hyp.get('fl_gamma', 2.5)

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer, clml_logger)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')


"""
TODO
Anchors,
    hyp['anchor_t'] = 4 let the AR<=4 => TODO check if valid 
    Ive reduced anchors to 2 per anchors: 2
Sampler : torch_weighted : WeightedRandomSampler
PP-YOLO bumps the batch size up from 64 to 192. Of course, this is hard to implement if you have GPU memory constraints.


******  DONT FORGET to delete cache files upon changing data  ************

python train.py --workers 8 --device 'cpu' --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'v7' --name yolov7 --hyp data/hyp.scratch.p5.yaml
--workers 8 --device cpu --batch-size 32 --data data/tir_od.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'v7' --name yolov7 --cache-images --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1
--workers 8 --device cpu --batch-size 32 --data data/tir_od.yaml --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weights 'v7' --name yolov7 --cache-images --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --multi-scale
--multi-scale training with resized image resolution not good for TIR
TRaining based on given model w/o prototype yaml by the --cfg

--workers 8 --device 0 --batch-size 16 --data data/coco_2_tir.yaml --img 640 640 --weights ./yolov7/yolov7.pt --name yolov7 --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 3 --linear-lr --noautoanchor

--workers 8 --device 0 --batch-size 16 --data data/tir_od.yaml --img 640 640 --weights ./yolov7/yolov7-tiny.pt --name yolov7 --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 3 --linear-lr --noautoanchor

===========================================================================
FT : you need the --cfg of arch yaml because nc-classes are changing 
--workers 8 --device 0 --batch-size 16 --data data/tir_od.yaml --img 640 640 --weights ./yolov7/yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --name yolov7 --hyp data/hyp.tir_od.tiny.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 3 --linear-lr


--workers 8 --device 0 --batch-size 16 --data data/tir_od.yaml --img 640 640 --weights ./yolov7/yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --name yolov7 --hyp hyp.tir_od.tiny_aug.yaml --adam --norm-type single_image_mean_std --input-channels 3 --linear-lr --epochs 2


--workers 8 --device 0 --batch-size 32 --data data/tir_od_center_roi_aug_list.yaml --img-size 640 --weights /mnt/Data/hanoch/tir_frames_rois/yolov7.pt --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od.tiny_aug_gamma_scaling_before_mosaic_rnd_scaling.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 100 --nosave --gamma-aug-prob 0.1 --cache-images


Overfit 640x640
tir_od_overfit.yaml
--workers 8 --device 0 --batch-size 32 --data data/tir_od_overfit.yaml --img-size 640 --weights /mnt/Data/hanoch/pretrained_coco_models/yolov7.pt --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od_v7_overfit.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 100 --nosave --gamma-aug-prob 0.1 --cache-images
--workers 8 --device 0 --batch-size 24 --data data/tir_od_overfit.yaml --img 640 640 --weights /mnt/Data/hanoch/pretrained_coco_models/yolov7.pt --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od.tiny_aug_gamma_scaling_before_mosaic_rnd_scaling.yaml --adam --linear-lr --norm-type single_image_percentile_0_1 --input-channels 1 --epochs 100 --gamma-aug-prob 0.1 --cache-images --image-weights --fl-gamma 1.5 --cosine-anneal --ohem-start-ep 10 --ohem-topk 0.7

# 3 classe renew yolov7999 list
--workers 8 --device 0 --batch-size 24 --data data/tir_od_center_roi_aug_list_train_cls.yaml --img 640 640 --weights /mnt/Data/hanoch/tir_frames_rois/yolov7.pt --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od.tiny_aug_gamma_scaling_before_mosaic_rnd_scaling_no_ota.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 100 --gamma-aug-prob 0.1 --cache-images --image-weights

--workers 8 --device 0 --batch-size 24 --data data/tir_od_center_roi_aug_list_train_cls.yaml --img 640 640 --weights /mnt/Data/hanoch/tir_frames_rois/yolov7.pt --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od.tiny_aug_gamma_scaling_before_mosaic_rnd_scaling_no_ota.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 100 --gamma-aug-prob 0.1 --cache-images --image-weights --loss-weight

#########################################################
Extended model for higher resolution  YOLO7E6
# --workers 8 --device 0 --batch-size 8 --data data/tir_od_center_roi_aug_list_full_res.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7-e6.pt --img-size [768, 1024] --cfg cfg/deploy/yolov7-e6.yaml --name yolov7e --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 2 --gamma-aug-prob 0.3 --cache-images --rect
# --workers 8 --device 0 --batch-size 8 --data data/tir_od_center_roi_aug_list_full_res.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7-e6.pt --img-size 1024 --cfg cfg/deploy/yolov7-e6.yaml --name yolov7e --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 10 --gamma-aug-prob 0.3 --cache-images
# --workers 1 --device 0 --batch-size 8 --data data/tir_od_center_roi_aug_list_full_res.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7-e6.pt --img-size 1024 --cfg cfg/deploy/yolov7-e6.yaml --name yolov7e --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 10 --gamma-aug-prob 0.3 --cache-images

# 1280 model 
--workers 8 --device 0 --batch-size 8 --data data/tir_od_center_roi_aug_list_full_res.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7-e6.pt --img-size 1024 --cfg cfg/deploy/yolov7-e6.yaml --name yolov7e --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 150 --gamma-aug-prob 0.3 --cache-images --project runs/train_7e

# union list with all Seq/TIff/Png

--workers 8 --device 0 --batch-size 12 --data data/tir_od_full_res.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7-e6.pt --img-size 1024 --cfg cfg/deploy/yolov7-e6.yaml --name yolov7e --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 150 --gamma-aug-prob 0.3 --cache-images --project runs/train_7e

Overfit ful_res
--workers 8 --device 0 --batch-size 8 --data data/tir_od_full_res_overfit.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7-e6.pt --img-size 1024 --cfg cfg/deploy/yolov7-e6.yaml --name yolov7e --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res_OVERFITTING.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 150 --gamma-aug-prob 0.3 --cache-images --project runs/train_7e

#CE with class weight
python -u ./yolov7/train.py --workers 8 --device 0 --batch-size 24 --data data/tir_od_center_roi_aug_list_train_cls_feb25.yaml --img 640 640 --weights /mnt/Data/hanoch/pretrained_coco_models/yolov7.pt --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od.tiny_aug_gamma_scaling_before_mosaic_rnd_scaling_no_ota.yaml --adam --linear-lr --norm-type single_image_percentile_0_1 --input-channels 1 --epochs 100 --gamma-aug-prob 0.1 --cache-images --image-weights --fl-gamma 1.5 --cosine-anneal --multi-class-no-multi-label --loss-weight


Trin yolov7 640 for 1024 
--workers 8 --device 0 --batch-size 16 --data data/tir_od_full_res.yaml --weights /mnt/Data/hanoch/tir_frames_rois/yolov7.pt --img-size 1024 1024 --cfg cfg/training/yolov7.yaml --name yolov7 --hyp hyp.tir_od.aug_gamma_scaling_before_mosaic_rnd_scaling_e6_full_res.yaml --adam --norm-type single_image_percentile_0_1 --input-channels 1 --linear-lr --epochs 100 --nosave --gamma-aug-prob 0.1 --cache-images --project runs/train_7_1024

class EMA_Clip(EMA):
    #Exponential moving average
    def _init_(self, mu, avg_factor=5):
        super()._init_(mu=mu)
        self.avg_factor = avg_factor

    def forward(self, x, last_average):
        if self.flag_first_time_passed==False:
            new_average = x
            self.flag_first_time_passed = True
        else:
            
            if x < self.avg_factor * last_average:
                new_average = self.mu * x + (1 - self.mu) * last_average
            else:
                new_average = self.mu * self.avg_factor * last_average + (1 - self.mu) * last_average
                
        return new_average
"""