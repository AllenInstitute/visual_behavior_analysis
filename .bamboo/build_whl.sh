set -eu

export PATH=${bamboo_build_working_directory}/miniconda/bin:$PATH
export HOME=${bamboo_build_working_directory}/.home
export TMPDIR=${bamboo_build_working_directory}
export CONDA_PATH_BACKUP=${CONDA_PATH_BACKUP:-$PATH}
export CONDA_PREFIX=${CONDA_PREFIX:-}
export CONDA_PS1_BACKUP=${CONDA_PS1_BACKUP:-}

source activate ${bamboo_build_working_directory}/.conda/conda_test_env
cd ${bamboo_build_working_directory}
python setup.py bdist_wheel --dist-dir build
source deactivate


