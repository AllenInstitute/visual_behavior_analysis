translator
==========

Pipeline for converting raw foraging2 output files into structured DataFrames.
This is typically the first step when working with raw behavioral data files
(pickled foraging2 output).

The translator pipeline has two stages:

1. ``foraging2`` — extracts raw data streams from the pickle file into a ``core_data`` dict
2. ``core`` — annotates and assembles the ``extended_trials`` DataFrame from core_data

foraging2
---------

extract
~~~~~~~

Main entry point: ``data_to_change_detection_core(data)`` converts a loaded
foraging2 pickle into the ``core_data`` dictionary.

.. automodule:: visual_behavior.translator.foraging2.extract
   :members:

extract_stimuli
~~~~~~~~~~~~~~~

.. automodule:: visual_behavior.translator.foraging2.extract_stimuli
   :members:

extract_images
~~~~~~~~~~~~~~

.. automodule:: visual_behavior.translator.foraging2.extract_images
   :members:

extract_movies
~~~~~~~~~~~~~~

.. automodule:: visual_behavior.translator.foraging2.extract_movies
   :members:

core
----

annotate
~~~~~~~~

Annotation functions applied to the assembled ``extended_trials`` DataFrame,
adding columns such as trial descriptions, response latency, and outcome labels.

.. automodule:: visual_behavior.translator.core.annotate
   :members:

foraging (legacy)
-----------------

Support for the original foraging (pre-foraging2) data format.

.. automodule:: visual_behavior.translator.foraging.extract
   :members:
