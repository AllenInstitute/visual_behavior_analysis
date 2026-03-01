ophys
=====

Core classes and analysis tools for two-photon calcium imaging data.

dataset
-------

VisualBehaviorOphysDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The central data object for a single ophys experiment. All data attributes are
lazily loaded on first access and cached in memory. Can be constructed from a
local cache directory or from an AllenSDK ``BehaviorOphysExperiment`` object.

.. automodule:: visual_behavior.ophys.dataset.visual_behavior_ophys_dataset
   :members:

stimulus_processing
~~~~~~~~~~~~~~~~~~~

Functions for parsing and formatting stimulus presentation tables, including
adding omitted stimulus rows and computing image presentation metadata.

.. automodule:: visual_behavior.ophys.dataset.stimulus_processing
   :members:

extended_stimulus_processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extended stimulus table utilities, adding columns such as ``prior_image_name``,
``post_image_name``, and change-aligned metadata.

.. automodule:: visual_behavior.ophys.dataset.extended_stimulus_processing
   :members:

cell_matching_dataset
~~~~~~~~~~~~~~~~~~~~~

Dataset class for working with cells matched across multiple imaging sessions
(cross-session cell identity tracking).

.. automodule:: visual_behavior.ophys.dataset.cell_matching_dataset
   :members:

response_analysis
-----------------

ResponseAnalysis
~~~~~~~~~~~~~~~~~

The ``ResponseAnalysis`` class extracts peri-event dF/F (or event) traces for
each cell, aligned to trial changes, stimulus onsets, or omissions. It produces
tidy DataFrames with one row per cell per event, including mean response,
baseline, full trace, and event metadata.

**Response window defaults:**

- Trials: [-5, 5] seconds around change time; mean response in [0, 0.5] s
- Stimulus presentations: [-0.5, 0.75] s around onset; mean response in [0, 0.5] s
- Omissions: [-5, 5] s around omission; mean response in [0, 0.75] s

.. automodule:: visual_behavior.ophys.response_analysis.response_analysis
   :members:

response_processing
~~~~~~~~~~~~~~~~~~~~

Low-level functions for extracting and aligning trace segments, computing
mean responses, and building response DataFrames.

.. automodule:: visual_behavior.ophys.response_analysis.response_processing
   :members:

cell_metrics
~~~~~~~~~~~~~

Functions for computing cell-level summary metrics from response DataFrames,
including preferred image identity, image selectivity, and omission responses.

.. automodule:: visual_behavior.ophys.response_analysis.cell_metrics
   :members:

utilities (response_analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: visual_behavior.ophys.response_analysis.utilities
   :members:

io
--

create_analysis_files
~~~~~~~~~~~~~~~~~~~~~~

Scripts for generating and saving analysis files (response DataFrames, cell metrics)
to a local cache directory.

.. automodule:: visual_behavior.ophys.io.create_analysis_files
   :members:

create_multi_session_df
~~~~~~~~~~~~~~~~~~~~~~~~

Functions for building multi-session DataFrames by concatenating per-experiment
response DataFrames across a set of experiments.

.. automodule:: visual_behavior.ophys.io.create_multi_session_df
   :members:

create_multi_session_mean_df
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: visual_behavior.ophys.io.create_multi_session_mean_df
   :members:

container_analysis
------------------

Utilities for aggregating and analyzing data at the container level
(a container is the set of sessions for one mouse in one imaging location).

.. automodule:: visual_behavior.ophys.container_analysis.utilities
   :members:

sync
----

Utilities for loading and processing sync files that align imaging frames
to behavioral events.

.. automodule:: visual_behavior.ophys.sync.process_sync
   :members:
