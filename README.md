braintv_behavior_piloting
==============================

Package for analyzing behavioral data from the BrainTV Visual Behavior Project.

## Quickstart

    pip install git+http://justink@stash.corp.alleninstitute.org/scm/~justink/braintv_behavior_piloting.git

## Installation

This package is designed to be installed using standard Python packaging tools. For example,

    python setup.py install

If you are using pip to manage packages and versions (recommended), you can also install using pip:

    pip install ./

If you are plan to contribute to the development of the package, I recommend installing in "editable" mode:

   pip install -e ./

This ensures that Python uses the current, active files in the folder (even while switching between branches).

## API

Here's a quick overview of each of the files:

- *analyze*: general purpose analysis functions
- *cohorts*: functions for getting information about training cohorts
- *core*: core functions (currently empty. will move some "utilities" here)
- *data*: manipulating and annotating data (especially trial dataframes)
- *devices*: accessing training devices
- *io*: reading & writing data
- *latest*: for syncing the latest data to a local folder
- *masks*: common/useful masks for filtering trials from the trial dataframe
- *metrics.classification*: sklearn-style metrics for behavior
- *metrics.session*: summary metrics for individual sessions
- *plotting*: plotting functions
- *utilities*: general purposes utilities

## Contributing

Pull requests are welcome. 

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Create a pull request
5. Tag `justink` to review

## Contributors:

- Doug Ollerenshaw - dougo@alleninstitute.org
- Justin Kiggins - justink@alleninstitute.org


## Additional Links

- [BrainTV Visual Behavior Project Page](http://confluence.corp.alleninstitute.org/display/CP/Brain+Observatory%3A+Visual+Behavior)
- [Details on Cohort Training](http://confluence.corp.alleninstitute.org/display/CP/_EXPERIMENTS)

