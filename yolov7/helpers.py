import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 root directory


def add_to_syspath():
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))


def remove_from_syspath():
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))

