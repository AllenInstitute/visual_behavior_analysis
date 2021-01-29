#!/bin/bash

set -eu

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p ${bamboo_build_working_directory}/miniconda
export PATH=${bamboo_build_working_directory}/miniconda/bin:$PATH
export HOME=${bamboo_build_working_directory}/.home
export TMPDIR=${bamboo_build_working_directory}
export CONDA_PATH_BACKUP=${CONDA_PATH_BACKUP:-$PATH}
export CONDA_PREFIX=${CONDA_PREFIX:-}
export CONDA_PS1_BACKUP=${CONDA_PS1_BACKUP:-}
conda create -y -v --prefix ${bamboo_build_working_directory}/.conda/conda_test_env python=${PYTHONVERSION}
source activate ${bamboo_build_working_directory}/.conda/conda_test_env
cd ${bamboo_build_working_directory}
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install git+https://github.com/AllenInstitute/allensdk.git@master
pip install .

source deactivate

