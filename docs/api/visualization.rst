visualization
=============

Plotting functions organized by data type and analysis level.

ophys
-----

summary_figures
~~~~~~~~~~~~~~~

Standard single-experiment summary plots: mean images, ROI masks, dF/F traces,
and response heatmaps.

.. automodule:: visual_behavior.visualization.ophys.summary_figures
   :members:

timeseries_figures
~~~~~~~~~~~~~~~~~~

Figures showing neural and behavioral timeseries (dF/F, running, licks, rewards)
aligned to behavioral events.

.. automodule:: visual_behavior.visualization.ophys.timeseries_figures
   :members:

experiment_summary_figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Figures summarizing response properties across all cells in a single experiment.

.. automodule:: visual_behavior.visualization.ophys.experiment_summary_figures
   :members:

population_summary_figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Population-level summary figures across sessions or containers.

.. automodule:: visual_behavior.visualization.ophys.population_summary_figures
   :members:

container_figures
~~~~~~~~~~~~~~~~~

Figures comparing data across sessions within a container
(same mouse, same imaging location).

.. automodule:: visual_behavior.visualization.ophys.container_figures
   :members:

platform_paper_figures
~~~~~~~~~~~~~~~~~~~~~~~

Figures generated for the Visual Behavior platform paper.

.. automodule:: visual_behavior.visualization.ophys.platform_paper_figures
   :members:

qc
--

Quality control plots for inspecting data at the session, experiment,
container, and single-cell levels.

plots
~~~~~

.. automodule:: visual_behavior.visualization.qc.plots
   :members:

session_plots
~~~~~~~~~~~~~

.. automodule:: visual_behavior.visualization.qc.session_plots
   :members:

experiment_plots
~~~~~~~~~~~~~~~~

.. automodule:: visual_behavior.visualization.qc.experiment_plots
   :members:

container_plots
~~~~~~~~~~~~~~~

.. automodule:: visual_behavior.visualization.qc.container_plots
   :members:

single_cell_plots
~~~~~~~~~~~~~~~~~

.. automodule:: visual_behavior.visualization.qc.single_cell_plots
   :members:

overview_plots
~~~~~~~~~~~~~~

.. automodule:: visual_behavior.visualization.qc.overview_plots
   :members:

dash_app
~~~~~~~~

Interactive Dash/Plotly web application for QC inspection. Launch with
``python -m visual_behavior.visualization.qc.dash_app.app``.

.. automodule:: visual_behavior.visualization.qc.dash_app.app
   :members:

extended_trials
---------------

Plots operating on the ``extended_trials`` DataFrame (behavioral sessions).

daily
~~~~~

Per-session behavioral summary plots.

.. automodule:: visual_behavior.visualization.extended_trials.daily
   :members:

mouse
~~~~~

Per-mouse training progress plots.

.. automodule:: visual_behavior.visualization.extended_trials.mouse
   :members:

utils
-----

Shared plotting utilities (color palettes, axis formatting, figure saving).

.. automodule:: visual_behavior.visualization.utils
   :members:
