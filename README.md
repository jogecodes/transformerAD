# transformerAD
Code for the paper "Anomaly-Based Intrusion Detection in IIoT Networks Using Transformer Models". 

The utils folder contains the implementation for the transformer class and all of its dependencies, along with some extra functions providing data loading and model saving support. The train-main.py and test-main.py files contain examples of model training and saving and AD execution, with saving of the results. The data-visualization.ipynb Jupyter Notebook includes the code for result evaluation and visualization.

Models will be saved in a /models/ directory, while its outputs will be kept in an /output/ directory. A /data/ directory must contain the processed datasets in its respective subfolders. 

This repository has been tested on the WUSTL-IIoT-2021 datasets. The code for the processing of both datasets in order for them to be in a format compatible with the transformer model can be found in the /dataset-processing directory. In order to reproduce the experiments presented on the paper:

1) Follow the instructions in the README file in /raw-data/ to download the WUSTL-IIoT dataset. Extract the CSV file in this same directory and keep the original name.
2) Execute the processing.ipynb Jupyter Notebook from the root directory. Note: All the cells must be executed.
3) Run the train-main.py script from the root directory. Hyperparameters can be tweaked.
4) Run the test-main.py script from the root directory. Testing batch size can be adjusted to memory requirements.
5) Use the data-visualization.ipynb Jupyter Notebook from the root directory to analyze the produced results.

Note that the train-main.py script will always generate a new numerated model subfolder. The 'model_name' variable in the test-main.py script must aim to the desired model. Similarly, running test-main.py will generate a new output subfolder relative to the used model. The 'output' variable in the data-visualization.ipynb Notebook must refer to the desired output as well.

Different datasets can be used with minimal adjustment.

