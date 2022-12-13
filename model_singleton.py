
from models.experimental import attempt_load
from utils.torch_utils import select_device

class ModelSingleton:
    _model = None
    _stride = None

    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, weights_path: str, device_name: str):
        device = select_device(device_name)
    
        # Load model
        self._model = attempt_load(weights_path, map_location=device)  # load FP32 model
        self._stride = int(self._model.stride.max())  # model stride

    def get_model(self):
        return self._model

    def get_stride(self):
        return self._stride
