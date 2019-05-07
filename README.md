Visual Behavior
==============================

Package for analyzing behavioral data from the BrainTV Visual Behavior Project.

## Before installing, it's recommended to set up a new Python environment:

For example, using Conda:

    conda create -n visual_behavior_analysis python=3.7

Then activate the environment:

    conda activate visual_behavior_analysis

## Quickstart

and install with pip:

    pip install git+http://stash.corp.alleninstitute.org/scm/vb/visual_behavior_analysis.git

## Installation

This package is designed to be installed using standard Python packaging tools. For example,

    python setup.py install

If you are using pip to manage packages and versions (recommended), you can also install using pip:

    pip install ./

If you are plan to contribute to the development of the package, I recommend installing in "editable" mode:

   pip install -e ./

This ensures that Python uses the current, active files in the folder (even while switching between branches).

To install from with in the AIBS local network from a whl using pip:
   
   pip install -i http://aibs-artifactory/artifactory/api/pypi/pypi-local/simple --trusted-host aibs-artifactory --extra-index-url https://pypi.org/simple visual_behavior==0.5.0.dev5


## To ensure that the newly created environment is visible in Jupyter:

Activate the environment:

    conda activate visual_behavior_analysis

Install ipykernel:

    pip install ipykernel

Register the environment with Jupyter:

    python -m ipykernel install --user --name visual_behavior_analysis

## Use

First, load up a Foraging2 output

``` Python
import pandas as pd
data = pd.read_pickle(PATH_TO_FORAGING2_OUTPUT_PKL)
```

Then, we create the "core" data structure: a dictionary with licks, rewards, trials, running, visual stimuli, and metadata.

``` Python
from visual_behavior.translator.foraging2 import data_to_change_detection_core

core_data = foraging2.data_to_change_detection_core(data)
```

Finally, we create an "extended" dataframe for use in generating trial-level plots and analysis.

``` Python
from visual_behavior.translator.core import create_extended_dataframe

extended_trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'],
)
```

## Contributing

Pull requests are welcome.

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Create a pull request
5. Tag `@dougollerenshaw`, `@matchings`, `@nicain` or `@nickponvert` to review

## Contributors:

- Christopher Mochizuki - chrim@alleninstitute.org
- Doug Ollerenshaw - dougo@alleninstitute.org
- Justin Kiggins - justink@alleninstitute.org
- Sahar Manavi - saharm@alleninstitute.org
- Nicholas Cain - nicholasc@alleninstitute.org
- Ryan Valenza - ryanv@alleninstitute.org
- Marina Garrett - marinag@alleninstitute.org
- Nick Ponvert - nick.ponvert@alleninstitute.org


## Additional Links

- [BrainTV Visual Behavior Project Page](http://confluence.corp.alleninstitute.org/display/CP/Brain+Observatory%3A+Visual+Behavior)
- [Details on Cohort Training](http://confluence.corp.alleninstitute.org/display/CP/_EXPERIMENTS)
