dimensionality_reduction
========================

Tools for dimensionality reduction and unsupervised clustering of neural population data.
Primarily operates on multi-session response DataFrames and GLM output.

clustering
----------

processing
~~~~~~~~~~

Spectral clustering of cells based on their response profiles or GLM weight vectors.
Uses scikit-learn's ``SpectralClustering`` and evaluates cluster quality via
silhouette score. Also includes utilities for computing pairwise distances and
integrating GLM-derived features.

.. automodule:: visual_behavior.dimensionality_reduction.clustering.processing
   :members:

plotting
~~~~~~~~

Cluster summary plots: heatmaps of cluster-averaged responses, UMAP embeddings
colored by cluster identity, and per-cluster response profiles.

.. automodule:: visual_behavior.dimensionality_reduction.clustering.plotting
   :members:

figures
~~~~~~~

Higher-level figure functions combining clustering results with metadata.

.. automodule:: visual_behavior.dimensionality_reduction.clustering.figures
   :members:

single_cell_plots
~~~~~~~~~~~~~~~~~

Per-cell plots for inspecting individual cell assignments within clusters.

.. automodule:: visual_behavior.dimensionality_reduction.clustering.single_cell_plots
   :members:

tca
---

Tensor component analysis (TCA) for decomposing population activity across
trials, neurons, and time.

processing
~~~~~~~~~~

.. automodule:: visual_behavior.dimensionality_reduction.tca.processing
   :members:

plotting
~~~~~~~~

.. automodule:: visual_behavior.dimensionality_reduction.tca.plotting
   :members:
