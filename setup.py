"""YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"""

import pathlib
import shlex
import subprocess
import warnings

from pkg_resources import parse_requirements
from setuptools import setup, find_packages

# Package version
VERSION = '1.0.0'

def version(base_version=VERSION, default_branch='main'):
    """Prepare SCM-aware package version string."""
    branch_name = subprocess.check_output(shlex.split("git rev-parse --abbrev-ref HEAD"), text=True).strip()
    if branch_name == default_branch:
        return base_version
    else:
        short_commit_sha = subprocess.check_output(shlex.split("git rev-parse --short HEAD"), text=True).strip()
        return f'{base_version}+g{short_commit_sha}'


# Packages to install
packages = find_packages(exclude=('tests', 'tests.*'))

# Parse dependencies from requirements.txt
install_requires = []
try:
    with pathlib.Path('requirements.txt').open() as requirements_txt:
        install_requires = [
            str(requirement) \
            for requirement in parse_requirements(requirements_txt)
        ]
except Exception as e:
    warnings.warn(f"Unable to parse requirements! {type(e).__name__}: {str(e)}")

# Extras
extras_require = {}
extras_require['all'] = list(set(
    dep for dep_list in extras_require.values() for dep in dep_list))  # magic: all except test

tests_require = []
extras_require['test'] = tests_require  # magic: test only

# Setup with metadata
setup(
    name='yolov7',
    version=version(),
    description='OneFormer: One Transformer to Rule Universal Image Segmentation',
    url='https://github.com/WongKinYiu/yolov7',
    author='WongKinYiu',
    author_email='',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require
)