data_access
===========

Utilities for loading, filtering, reformatting, and querying Visual Behavior data.
This subpackage is the primary interface for retrieving experiment and session tables,
loading individual experiments via the AllenSDK cache or LIMS, and computing
pre/post-condition metrics.

loading
-------

High-level functions for loading experiment and session tables, individual datasets,
and derived data files (e.g., multi-session DataFrames).

.. automodule:: visual_behavior.data_access.loading
   :members:

filtering
---------

Functions for filtering experiment and session tables by cre line, imaging depth,
session type, project code, and other metadata fields.

.. automodule:: visual_behavior.data_access.filtering
   :members:

reformat
--------

Functions for reshaping and reformatting tables — e.g., renaming columns,
adding derived columns, converting session types to human-readable labels.

.. automodule:: visual_behavior.data_access.reformat
   :members:

utilities
---------

General-purpose utility functions used across the data_access subpackage.

.. automodule:: visual_behavior.data_access.utilities
   :members:

from_lims
---------

Functions for querying the Allen Institute LIMS database directly.
Requires on-premises access and LIMS environment variables to be set.

.. automodule:: visual_behavior.data_access.from_lims
   :members:

from_lims_utilities
-------------------

Lower-level LIMS query helpers.

.. automodule:: visual_behavior.data_access.from_lims_utilities
   :members:

processing
----------

Functions for computing derived quantities from raw data tables
(e.g., adding running statistics, pupil area metrics).

.. automodule:: visual_behavior.data_access.processing
   :members:

pre_post_conditions
-------------------

Functions for computing metrics comparing neural or behavioral responses
in familiar vs. novel session conditions (pre/post learning).

.. automodule:: visual_behavior.data_access.pre_post_conditions
   :members:
