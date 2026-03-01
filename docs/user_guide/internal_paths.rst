.. _internal_paths:

Internal Filesystem Paths
=========================

.. warning::

   Many functions in this codebase contain **hardcoded Allen Institute filesystem
   paths** (UNC paths such as ``\\\\allen\\programs\\braintv\\...`` or Linux
   equivalents ``/allen/programs/braintv/...``).  These paths are only accessible
   from **within the Allen Institute network**.

   External users and Code Ocean capsule users will receive ``FileNotFoundError``
   or ``OSError`` when these defaults are used.

Affected areas
--------------

The following modules fall back to internal paths when no explicit
``cache_dir`` / ``analysis_cache_dir`` argument is supplied:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - Default path behaviour
   * - ``VisualBehaviorOphysDataset``
     - Uses ``/allen/aibs/informatics/swdb2018/visual_behavior`` (Linux) or
       ``\\\\allen\\aibs\\informatics\\swdb2018\\visual_behavior`` (Windows)
       when ``cache_dir=None``.
   * - ``ResponseAnalysis``
     - Uses ``//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/
       visual_behavior_production_analysis`` when ``analysis_cache_dir=None``.
   * - ``visual_behavior.data_access.loading``
     - Several loader functions query the Allen LIMS database and/or read
       from internal network shares.  These require the environment variables
       ``LIMS_DBNAME``, ``LIMS_USER``, ``LIMS_HOST``, ``LIMS_PASSWORD``, and
       ``LIMS_PORT`` to be set correctly.
   * - ``visual_behavior.ophys.io.create_analysis_files``
     - The ``__main__`` block and ``cache_dir`` default point to the
       production analysis share.

How to use this package without internal access
-----------------------------------------------

Option 1 — AllenSDK public cache (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Allen Institute publishes the Visual Behavior Ophys dataset through the
`AllenSDK <https://allensdk.readthedocs.io/en/latest/visual_behavior_ophys.html>`_.
This requires no internal access and provides a standard
``BehaviorOphysExperiment`` object that is compatible with
``ResponseAnalysis``.

.. code-block:: python

   from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

   cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='/my/local/cache')
   experiment = cache.get_behavior_ophys_experiment(ophys_experiment_id=123456789)

   from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

   analysis = ResponseAnalysis(
       dataset=experiment,
       analysis_cache_dir='/my/local/analysis_cache',  # <-- provide your own path
   )

Option 2 — ``VisualBehaviorOphysDataset`` with a local cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a locally-mirrored copy of the dataset files:

.. code-block:: python

   from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset

   dataset = VisualBehaviorOphysDataset(
       experiment_id=123456789,
       cache_dir='/path/to/your/local/data',   # <-- provide your own path
   )

Always pass ``cache_dir`` and ``analysis_cache_dir`` explicitly to avoid
hitting the Allen Institute defaults.
