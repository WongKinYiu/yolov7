import logging
import mlflow
import os
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv
from typing import Dict, Union

from mlflow.tracking import MlflowClient
import logging

load_dotenv()


class MLFlowLogger():
    def __init__(self):
        try:
            mlflow.set_tracking_uri(os.environ['LOAD_BALANCER_URI'])
        except KeyError as e:
            raise Exception('LOAD_BALANCER_URI environment variable was not set') from e
        self.client = MlflowClient()

    def register_model(self, ):
        """ Register wrapped model class to mlflow based on specific rule"""
        # https://medium.com/analytics-vidhya/packaging-your-pytorch-model-using-mlflow-894d62dd8d3
        return

    def retrieve_model(self, ):
        """ Retrieve registered model from mlflow model registry """
        return

    def log_run(self, experiment_name: str, run_name: str = None, params: Dict[str, str] = None,
                metrics: Dict[str, float] = None,
                artifacts: str = None, images: Dict[str, Union[np.ndarray, Image.Image]] = None):
        """ Start mlflow experiment run and log results """
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if artifacts:
                mlflow.log_artifacts(artifacts)
            if images:
                for file, image in images.items():
                    mlflow.log_image(image, file)
        return

    def download_mlflow_artifacts(self, version: str, default: bool):
        """ Download artifacts from specific run, either default or trained experiment runs """
        if default:
            directory = f"mlflow-artifacts/default/{version}/"
        else:
            directory = f"mlflow-artifacts/trained/{version}/"

        if os.path.exists(directory):
            logging.info("Artifacts allready exists locally")
            return directory
        else:
            os.makedirs(directory)
            if default:
                exp = mlflow.get_experiment_by_name("Default Models")
            else:
                exp = mlflow.get_experiment_by_name("Trained Models")
            query = f"tags.mlflow.runName = '{version}'"
            run_id = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=query)['run_id'].values[0]
            return self.client.download_artifacts(run_id=run_id, path="", dst_path=directory)


if __name__ == '__main__':
    # Default model json including default location yolo model artifacts to log
    with open('default-model.json') as json_file:
        m = json.load(json_file)

    logger = MLFlowLogger()
    # Make sure to run "python download-yolo.py" first to download the default yolo model artifacts in model.json
    logger.log_run(experiment_name="Default Models", run_name=m['version'], params=m['artifacts'],
            artifacts=f"artifacts/{m['version']}/")

    # Download mlflow artifacts
    logger.download_mlflow_artifacts(version=m['version'], default=True)

