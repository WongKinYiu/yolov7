"""
This script contains the utility functions for the mlflow callbacks
"""

import mlflow
from pathlib import Path
import logging
LOGGER = logging.getLogger(__name__)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]


def colorstr(*input):
    """
    This function colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, 
    i.e.  colorstr('blue', 'hello world')
    """

    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def on_pretrain_routine_end(info_dict):
    """
    This function creates the mlflow experiment for the training and log some basic information's,

    Parameters
    ----------
    info_dict : dictionary
        python dictionary with following fields,
        input size, number of classes, layers, class names
    
        TODO: add other fields which are necessary
    """

    global mlflow, _mlflow, _run_id, _expr_name

    mlflow_location = 'http://127.0.0.1:5000' 
    mlflow.set_tracking_uri(mlflow_location)

    _expr_name = 'Yolov7'
    experiment = mlflow.get_experiment_by_name(_expr_name)
    if experiment is None:
        mlflow.create_experiment(_expr_name)
    mlflow.set_experiment(_expr_name)

    prefix = colorstr('MLFlow: ')
    try:
        _mlflow, mlflow_active_run = mlflow, None if not mlflow else mlflow.start_run()
        if mlflow_active_run is not None:
            _run_id = mlflow_active_run.info.run_id
            LOGGER.info(f'{prefix}Using run_id({_run_id}) at {mlflow_location}')
    except Exception as err:
        LOGGER.error(f'{prefix}Failing init - {repr(err)}')
        LOGGER.warning(f'{prefix}Continuing without mlflow')
        _mlflow = None
        mlflow_active_run = None

    _mlflow.log_params(info_dict)


def on_fit_epoch_end(results, maps, names, epoch):
    """
    This function logs the epoch wise results

    Parameters
    ----------
    results : list
        list of mean values and validation losses calculated at the end of every epoch
    maps : list
        class wise average precision
    names : list
        list of class names
    epoch : int
        number of epoch
    """
    
    class_wise_ap = {}
    for i, c in enumerate(maps):
        class_wise_ap[names[i]] = c

    metrics_dict = {
            "mp": results[0],
            "mr": results[1],
            "mAP5": results[2],
            "mAP95": results[3],
            "vallossbox": results[4],
            "vallossobj": results[5],
            "vallosscls": results[6]

    }

    metrics_dict.update(class_wise_ap)

    _mlflow.log_metrics(metrics=metrics_dict, step=epoch)


def on_model_save(save_dir):
    """
    This function saves the last model

    Parameters
    ----------
    save_dir : str
        local path to the last.pt
    """
    _mlflow.log_artifact(save_dir, 'last')


def on_train_end(save_dir):
    """
    This function saves the best model and register it on model registry

    Parameters
    ----------
    save_dir : str
        local path to the best.pt
    """
    _mlflow.log_artifact(save_dir, 'best')
    model_uri = f'runs:/{_run_id}/'
    _mlflow.register_model(model_uri, _expr_name)

