import os
from typing import Dict, Any
import yaml

home_dir = os.getenv("HOME")
persistent_dir = os.path.join(home_dir, "Tensorleap", 'ALBERTqa')
root = os.path.abspath(os.path.dirname(__file__))

def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    file_path = os.path.join(root, 'project_config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

CONFIG = load_od_config()
with open(os.path.join(os.path.dirname(root),CONFIG['HYP'])) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
print(CONFIG)
with open(os.path.join(os.path.dirname(root),CONFIG['DATA'])) as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)

with open(os.path.join(os.path.dirname(root), CONFIG['DATA_TEST'])) as f:
    data_test_dict = yaml.load(f, Loader=yaml.SafeLoader)
