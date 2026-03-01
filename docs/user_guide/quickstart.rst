Quickstart
==========

Loading behavior data (foraging2 format)
-----------------------------------------

Raw behavioral sessions are stored as pickled foraging2 output files. The translator
pipeline converts these into structured DataFrames:

.. code-block:: python

   import pandas as pd
   from visual_behavior.translator.foraging2 import data_to_change_detection_core
   from visual_behavior.translator.core import create_extended_dataframe

   # Load raw foraging2 pickle
   data = pd.read_pickle('/path/to/session.pkl')

   # Convert to core data structure (dict with licks, rewards, trials, running, stimuli, metadata)
   core_data = data_to_change_detection_core(data)

   # Build extended trials dataframe with per-trial metrics and annotations
   extended_trials = create_extended_dataframe(
       trials=core_data['trials'],
       metadata=core_data['metadata'],
       licks=core_data['licks'],
       time=core_data['time'],
   )

Loading ophys data via AllenSDK cache (public)
-----------------------------------------------

.. code-block:: python

   from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
   from visual_behavior.data_access import loading

   cache_dir = '/path/to/local/cache'
   cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)

   # Get experiment and session tables
   ophys_experiment_table = loading.get_filtered_ophys_experiment_table()
   ophys_session_table = loading.get_filtered_ophys_session_table()

   # Load a single experiment
   experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=123456789)

Loading ophys data via the legacy dataset class
------------------------------------------------

For internal Allen Institute users with LIMS access, the ``VisualBehaviorOphysDataset``
class provides direct loading from the institute's data storage:

.. code-block:: python

   from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset

   dataset = VisualBehaviorOphysDataset(experiment_id=123456789, cache_dir='/path/to/cache')

   # Access data as attributes (lazily loaded)
   dff_traces     = dataset.dff_traces        # DataFrame: cells x time
   events         = dataset.events            # DataFrame: detected neural events
   stimulus_table = dataset.stimulus_presentations  # DataFrame: one row per stimulus flash
   running_speed  = dataset.running_speed     # DataFrame: speed over time
   licks          = dataset.licks             # DataFrame: lick times
   rewards        = dataset.rewards           # DataFrame: reward times
   metadata       = dataset.metadata          # dict: session/experiment metadata

Computing response dataframes
------------------------------

``ResponseAnalysis`` extracts peri-event dF/F traces and computes mean responses
for each cell across trials, stimulus presentations, and omissions:

.. code-block:: python

   from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

   # Pass a dataset object (VisualBehaviorOphysDataset or BehaviorOphysExperiment)
   ra = ResponseAnalysis(dataset, use_events=False, use_extended_stimulus_presentations=True)

   # Available response dataframe types
   trials_df    = ra.get_response_df('trials_response_df')
   stimulus_df  = ra.get_response_df('stimulus_response_df')
   omission_df  = ra.get_response_df('omission_response_df')

   # Each df has columns: cell_specimen_id, mean_response, trace, image_name, etc.

Computing cell-level metrics
-----------------------------

.. code-block:: python

   from visual_behavior.ophys.response_analysis import cell_metrics

   # Preferred image for each cell (image evoking max mean response)
   pref_images = cell_metrics.get_pref_image_for_cell_specimen_ids(stimulus_df)

   # Add preferred image label to response df
   stimulus_df = cell_metrics.add_pref_image(stimulus_df, pref_images)
