Installation
============

Requirements
------------

- Python 3.8+
- A conda environment is strongly recommended

Create a new environment
------------------------

.. code-block:: bash

   conda create -n visual_behavior python=3.9
   conda activate visual_behavior

Install the package
-------------------

For read-only use (e.g., running analyses):

.. code-block:: bash

   pip install git+https://github.com/AllenInstitute/visual_behavior_analysis.git

For development (editable install, changes to source are reflected immediately):

.. code-block:: bash

   git clone https://github.com/AllenInstitute/visual_behavior_analysis.git
   cd visual_behavior_analysis
   pip install --upgrade pip setuptools
   pip install -e .

To include development dependencies (testing, linting):

.. code-block:: bash

   pip install -e ".[DEV]"

Register environment with Jupyter
----------------------------------

.. code-block:: bash

   conda activate visual_behavior
   pip install ipykernel
   python -m ipykernel install --user --name visual_behavior

Data access
-----------

Most ophys functionality requires either:

- Access to the Allen Institute internal LIMS database (on-premises users only), configured via environment variables ``LIMS_DBNAME``, ``LIMS_USER``, ``LIMS_HOST``, ``LIMS_PASSWORD``, ``LIMS_PORT``
- Or the public `AllenSDK cache <https://allensdk.readthedocs.io/en/latest/visual_behavior_ophys.html>`_, which can be set up without internal access

See :doc:`quickstart` for examples of both access patterns.

.. note::

   Many functions contain hardcoded Allen Institute filesystem paths that are
   unreachable from outside the Allen network.  See :doc:`internal_paths` for
   a full list and guidance on providing your own paths.
