import platform
import os

if platform.system() == 'Linux':
    basepath = "/allen/programs/braintv/workgroups/neuralcoding/Behavior/Data"
else:
    basepath = r"\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data"

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
