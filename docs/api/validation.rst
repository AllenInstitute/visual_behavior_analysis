validation
==========

Functions for validating data integrity at multiple levels of the pipeline,
from raw trial logs through to SDK-loaded experiments.

core
----

Core validation functions shared across validation modules.

.. automodule:: visual_behavior.validation.core
   :members:

trials
------

Validation checks on trial DataFrames (e.g., trial counts, outcome consistency,
timing sanity checks).

.. automodule:: visual_behavior.validation.trials
   :members:

extended_trials
---------------

Validation of the extended trials DataFrame, checking derived columns for
consistency with raw trial data.

.. automodule:: visual_behavior.validation.extended_trials
   :members:

foraging2
---------

Validation specific to the foraging2 data format.

.. automodule:: visual_behavior.validation.foraging2
   :members:

sdk
---

Validation checks for AllenSDK-loaded ``BehaviorOphysExperiment`` objects.

.. automodule:: visual_behavior.validation.sdk
   :members:

qc
--

Data quality control checks used in the QC pipeline.

.. automodule:: visual_behavior.validation.qc
   :members:
