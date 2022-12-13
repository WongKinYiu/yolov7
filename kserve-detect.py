import argparse
import os
import kserve

from kserve_wrapper.predictor import YoloV7Model

DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = YoloV7Model(args.model_name)
    model.load()
    kserve.ModelServer().start([model])
