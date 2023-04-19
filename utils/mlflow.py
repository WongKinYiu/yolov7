from pathlib import Path
import logging
LOGGER = logging.getLogger(__name__)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]

import mlflow

def on_pretrain_routine_end(nc, names, epochs, batch_size, img_size, initial_weights):
    """
    This callback function log basic model parameters
    Parameters
    ----------
    nc : int
        number of classes
    
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
        LOGGER.warning(f'{prefix}Continuining without Mlflow')
        _mlflow = None
        mlflow_active_run = None

    _mlflow.log_params({"nc":nc, "names":names, "epochs": epochs, "batch_size":batch_size, "img_size":img_size, "initial_weights":initial_weights})


def on_fit_epoch_end(results, epoch):
    """
    This callback function log basic model parameters

    Parameters
    ----------
    results : list
        list of matrices 
    epoch : int
        epoch number
    """
    metrics_dict = {
            "mp": results[0],
            "mr": results[1],
            "mAP5": results[2],
            "mAP95": results[3],
            "vallossbox": results[4],
            "vallossobj": results[5],
            "vallosscls": results[6],
    }

    _mlflow.log_metrics(metrics=metrics_dict, step=epoch)


def on_model_save(model_last):
    """
    This callback function log last check point

    Parameters
    ----------
    model_last : str
        local path to last.pt
    """
    _mlflow.log_artifact(model_last)


def on_train_end(save_dir, model_best):
    """
    This callback function log the best model check point

    Parameters
    ----------
    save_dir : str
        path to save model weights
    model_best : str
        local path to best.pt
    """

    # Save best model
    _mlflow.log_artifact(model_best)
    model_uri = f'runs:/{_run_id}/'

    # Save model
    _mlflow.register_model(model_uri, _expr_name)

    _mlflow.pyfunc.log_model(artifact_path=_expr_name,
                                code_path=[str(ROOT.resolve())],
                                artifacts={'model_path': str(save_dir)},
                                python_model=_mlflow.pyfunc.PythonModel())
