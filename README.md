# transformerAD
Code for the paper "Anomaly-Based Intrusion Detection in IIoT Networks Using Transformer Models". 

The utils folder contains the implementation for the transformer class and all of its dependencies, along with some extra functions providing data loading and model saving support. The train-main.py and test-main.py files contain examples of model training and saving and AD execution, with saving of the results. The data-visualization.ipynb Jupyter Notebook includes the code for result evaluation and visualization.

Models will be saved in the /models directory, while its outputs will be kept in the /output directory. the /data directory must contain the used datasets. 

This repository has been tested on the UNSW-NB15 and on the WUSTL-IIoT-2021 datasets. The code for the processing of both datasets in order for them to be in a format compatible with the transformer model can be found in the /dataset-processing directory.
