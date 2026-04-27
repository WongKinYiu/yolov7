#!/usr/bin/env python3
"""Compatibility checker entrypoint for DLA-exported YOLOv7-seg ONNX."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / 'seg' / 'tools' / 'check_dla_graph.py'
    if not script.exists():
        raise SystemExit(f'checker script not found: {script}')
    # Forward argv to the canonical checker implementation.
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
