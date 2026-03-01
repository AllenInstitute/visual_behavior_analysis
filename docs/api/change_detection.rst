change_detection
================

Behavioral analysis tools for the change detection task. Functions here operate
on trial-level DataFrames to compute performance metrics, running statistics,
lick patterns, and trial-type masks.

trials
------

session_metrics
~~~~~~~~~~~~~~~

Session-level behavioral performance metrics computed from trial DataFrames,
including hit rate, false alarm rate, d-prime, and reward rate.

.. automodule:: visual_behavior.change_detection.trials.session_metrics
   :members:

masks
~~~~~

Boolean mask functions for subsetting trial DataFrames by trial type
(go, catch, aborted) or outcome (hit, miss, false alarm, correct reject).

.. automodule:: visual_behavior.change_detection.trials.masks
   :members:

summarize
~~~~~~~~~

Functions for summarizing trial DataFrames into session-level summary statistics.

.. automodule:: visual_behavior.change_detection.trials.summarize
   :members:

validation
~~~~~~~~~~

Trial-level validation functions for checking data integrity.

.. automodule:: visual_behavior.change_detection.trials.validation
   :members:

running
-------

.. automodule:: visual_behavior.change_detection.running.metrics
   :members:
