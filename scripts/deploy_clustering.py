
import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
from simple_slurm import Slurm

# define the conda environment
conda_environment = 'vba'

# define the job record output folder
job_dir = r"//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/clustering"
# make the job record location if it doesn't already exist
os.mkdir(job_dir) if not os.path.exists(job_dir) else None

# env path
python_path = os.path.join(
    os.path.expanduser("~"),
    'anaconda3',
    'envs',
    conda_environment,
    'bin',
    'python'
)

# create slurm instance
slurm = Slurm(
    cpus_per_task=8,
    job_name='glm_clustering',
    mem = '5g',
    time = '1:30:00',
    partition = 'braintv',
    output=f'{job_dir}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
)

methods = ['kmeans', 'discretize']
metrics = ['braycurtis', 'canberra', 'chebyshev',
           'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
           'hamming', 'jaccard', 'kulsinski',
           'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
            'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
n_clusters = int(35)
n_boots = int(40)

for method in methods:
    for metric in metrics:
        print('running ' + method + ' ' + metric)
        slurm.sbatch('{} ../scripts/cluster_glm.py --method {} --metric {} --n_clusters {} --n_boots {}'.format(
            python_path,
            method,
            metric,
            n_clusters,
            n_boots,
            )
            # slurm.sbatch('{} //home/iryna.yavorska/code/vba-iryna/scripts/test.py')
        )
