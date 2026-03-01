Key Data Structures
===================

This page describes the core data structures used throughout the package.

core_data (dict)
-----------------

Produced by ``data_to_change_detection_core()``. A dictionary with keys:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Key
     - Description
   * - ``trials``
     - DataFrame with one row per trial. Includes trial type, outcome, change image, lick times, reward times.
   * - ``licks``
     - DataFrame with one row per lick event (timestamps).
   * - ``rewards``
     - DataFrame with one row per reward delivery (timestamps, volume).
   * - ``running``
     - DataFrame with running speed over time (timestamps, speed in cm/s).
   * - ``stimuli``
     - DataFrame describing stimulus presentations (image identity, onset/offset times).
   * - ``metadata``
     - Dict of session-level metadata (mouse ID, session type, rig, date, etc.).
   * - ``time``
     - Array of master timestamps for synchronization.

extended_trials (DataFrame)
-----------------------------

Produced by ``create_extended_dataframe()``. One row per trial, extending the base
``trials`` dict with derived behavioral metrics:

- ``response_latency``: time from change to first lick (seconds)
- ``response_binary``: whether the mouse responded on the trial
- ``trial_type``: ``go``, ``catch``, or ``aborted``
- ``hit``, ``miss``, ``false_alarm``, ``correct_reject``: boolean outcome columns
- ``d_prime``: session-level discriminability (scalar, same value across all rows)
- ``reward_rate``: rolling reward rate at time of trial

stimulus_response_df / trials_response_df (DataFrame)
-------------------------------------------------------

Produced by ``ResponseAnalysis.get_response_df()``. One row per cell per event:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - ``cell_specimen_id``
     - Unique cell identifier
   * - ``mean_response``
     - Mean dF/F (or event amplitude) in a post-stimulus window (default 500 ms)
   * - ``baseline_response``
     - Mean dF/F in a pre-stimulus window
   * - ``trace``
     - Full peri-event trace (array), length determined by window size
   * - ``trace_timestamps``
     - Timestamps for each point in ``trace``
   * - ``image_name``
     - Identity of the stimulus image (stimulus_response_df only)
   * - ``is_change``
     - Whether this was a change image presentation
   * - ``omitted``
     - Whether the stimulus was omitted
   * - ``licked``
     - Whether the mouse licked in the response window
   * - ``rewarded``
     - Whether a reward was delivered

VisualBehaviorOphysDataset attributes
---------------------------------------

Key attributes of the dataset object (all lazily loaded):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Attribute
     - Description
   * - ``dff_traces``
     - DataFrame (cells × time) of dF/F fluorescence traces
   * - ``events``
     - DataFrame of inferred neural events (from L0 event detection)
   * - ``cell_specimen_ids``
     - Array of unique cell identifiers
   * - ``roi_masks``
     - Dict mapping cell_specimen_id to 2D boolean ROI mask
   * - ``stimulus_presentations``
     - DataFrame of all stimulus flashes with image identity and timing
   * - ``trials``
     - DataFrame of behavioral trials
   * - ``running_speed``
     - DataFrame of running speed (timestamps, speed in cm/s)
   * - ``licks``
     - DataFrame of lick events
   * - ``rewards``
     - DataFrame of reward deliveries
   * - ``metadata``
     - Dict of session and experiment metadata
   * - ``ophys_timestamps``
     - Array of timestamps for each imaging frame
