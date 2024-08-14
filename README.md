# ScanOFC: Statistical framework for Clustering with Alignment and Network inference of Omic Fold Changes
A Python library containing tools for inference of multivariate omic fold changes from the data, for their subsequent clustering with alignment, and inference and visualisation of a network. Here is an overview of the main files:

- ### scanofc.py
Main script, contains 3 classes: FoldChanges, Clustering and NetworkInference. 

- ### simulation_examples.ipynb
A Jupyter notebook containing examples from simulation studies showcasing frequently observed patterns and some of the potential interesting outcomes.

- ### simulation_study_1.py
Main script of the first series of simulation studies focusing on the choice of distance and clustering algorithm.

- ### simulation_study_2.py
Main script of the second series of simulation studies focusing on the effect of alignment, and two clustering alternatives: stochastic block model inference and clustering of the coordinates of the UMAP projection of the distance matrix.

- ### scanofc_clustering_example.ipynb
A Jupyter notebook comparing the results obtained with the joint clustering with alignment framework implemented in ScanOFC with those obtained with spectral clustering. The methods are applied on a real dataset (LINAC).

- ### scanofc_tutorial.ipynb
A Jupyter notebook demonstrating how to use ScanOFC on two real datasets.

- ### scanofc_suppl_functions.py
Supplementary functions used in the tutorial.



