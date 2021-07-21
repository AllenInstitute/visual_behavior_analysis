Allen Institute Visual Behavior Analysis package
==============================

This repository contains code for analyzing behavioral data from the Allen Brain Observatory: Visual Behavior 2P Project.

This code is an important part of the internal Allen Institute code base and we are actively using and maintaining it. Issues are encouraged, but because this tool is so central to our mission pull requests might not be accepted if they conflict with our existing plans.

## Before installing, it's recommended to set up a new Python environment:

For example, using Conda:

    conda create -n visual_behavior_analysis python=3.7

Then activate the environment:

    conda activate visual_behavior_analysis

## Quickstart

and install with pip (Allen Institute internal users only):

    pip install git+https://github.com/AllenInstitute/visual_behavior_analysis.git
    
## Installation

This package is designed to be installed using standard Python packaging tools. For example,

    python setup.py install

If you are using pip to manage packages and versions (recommended), you can also install using pip:

    pip install ./

If you are plan to contribute to the development of the package, I recommend installing in "editable" mode:

    pip install -e ./

This ensures that Python uses the current, active files in the folder (even while switching between branches).


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

core_data = data_to_change_detection_core(data)
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

## Testing

Before committing and/or submitting a pull request, it is ideal to run tests.  

Tests are currently run against Python 3.6.12 and 3.7.7 on github using CircleCI. You can replicate those tests locally as follows:

**Creating test virtual environments**

    CD {your local VBA directory}
    conda create -n VBA_test_36 python=3.6.12
    conda activate VBA_test_36
    pip install .[DEV]

Then deactivate VBA_test_36 to create the 3.7 virtual environment:

    conda create -n VBA_test_37 python=3.7.7
    conda activate test_37
    pip install .[DEV]

**Basic testing (external users):**
Baseline tests consist of tests that can be run from outside of the Allen Institute and do not require access to any internal databases such as LIMS.  The `not onprem` argument will skip all tests that can only be run on internal Allen Institute servers and are marked as onprem.  To run these tests, do the following: 

    CD {your local VBA directory}
    conda activate VBA_test_36
    pytest -m "not onprem" 


**On Premises Testing + Basic testing (internal Allen Institute Users):**
Some tests may only be run on premises (at the Allen Institute) because they must access our internal databases such as LIMS. For internal Allen Institute users, the call to pytest could be called without an onprem argument, which would run ALL tests. To run these tests, do the following: 

    CD {your local VBA directory}
    conda activate VBA_test_36
    pytest 

**Linting / Circle CI Testing (all users):**

CircleCI also tests that all files meet Pep 8 style requirements using the Flake8 module - a process referred to as 'linting'. Linting can be performed locally before commiting using Flake8 as follows:

    flake8 {FILE_TO_CHECK}

**Running a subset of tests:**
You can run a subset of test by doing the following

All tests in a sub directory:

    CD {subfolder of VBA that contains the tests you'd like to run}
    conda activate VBA_test_36
    pytest {add -m "not onprem" as necessary}

All test in a single .py file:

    CD {subfolder of VBA that contains the file with the tests you'd like to run}
    conda activate VBA_test_36
    pytest fileWithTests.py  {add -m "not onprem" as necessary}


## Contributing

Pull requests are welcome.

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Create a pull request
5. Tag `@dougollerenshaw`, `@matchings` to review

## Contributors:

- Nicholas Cain - @nicain
- Marina Garrett - marinag@alleninstitute.org, @matchings
- Nile Graddis - nileg@alleninstitute.org, @nilegraddis
- Justin Kiggins - @neuromusic
- Jerome Lecoq - jeromel@alleninstitute.org, @jeromelecoq
- Sahar Manavi - saharm@alleninstitute.org, @saharmanavi
- Nicholas Mei - nicholas.mei@alleninstitute.org, @njmei
- Christopher Mochizuki - chrism@alleninstitute.org, @mochic
- Doug Ollerenshaw - dougo@alleninstitute.org, @dougollerenshaw
- Natalia Orlova - nataliao@alleninstitute.org, @nataliaorlova
- Jed Perkins - @jfperkins
- Alex Piet - alex.piet@alleninstitute.org, @alexpiet
- Nick Ponvert - @nickponvert
- Kate Roll - kater@alleninstitute.org, @downtoncrabby
- Ryan Valenza - @ryval
- Farzaneh Najafi - farzaneh.najafi@alleninstitute.org
- Iryna Yavorska - iryna.yavorska@alleninstitute.org


## Additional Links

- [AllenSDK](https://github.com/AllenInstitute/AllenSDK)
- [BrainTV Visual Behavior Project Page](http://confluence.corp.alleninstitute.org/display/CP/Brain+Observatory%3A+Visual+Behavior)
- [Details on Cohort Training](http://confluence.corp.alleninstitute.org/display/CP/_EXPERIMENTS)


