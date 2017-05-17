import platform

if platform.system() == 'Linux':
    basepath = "/data/neuralcoding/Behavior/Data"
else:
    basepath = r"\\aibsdata\neuralcoding\Behavior\Data"

import os
project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
