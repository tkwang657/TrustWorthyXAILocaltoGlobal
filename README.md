A project repository for TrustworthyAI local to global inference
Important notes:
The code for the Influence function is heavily re-adapted closely following this linked repository as a proof of concept implementation. It can be found within utilities/influence https://github.com/nimarb/pytorch_influence_functions

The TabNet model has also been re-adapted from the original PyTorch implementation for our custom usage. It can be found within utilities/models/pytorch_tabnet https://pypi.org/project/pytorch-tabnet/

Original dataset is the 2024 dataset, can be downloaded from link in paper.

1. The dataexplorer.ipynb file explores the original 12M+ rows dataset. However, the actual training and experiment is performed on a subset of 500k rows, which can be generated via the same notebook. 
For the sake of succinctness, the 500k sample dataset is provided separately here:

2. The "tabnettrain + influence.ipynb" file trains the TABNET model, computes the influence functions, and finishes both the clustering and tree building. (I trained it on the cluster with 2x16gb CPU cores and 1x GPU). 

For the rest of the notebook, it is crucial to run all the initial data processing cells until the ##Training## section. 

3. Alternative to training: preloaded modelcheckpoints are provided in training/models__ which can be loaded in the appropriate cell within the Jupyter notebook. Search for the headline "Pick an Epoch (Reload model here if it crashes during influence or if you do not want to train from scratch)"". By default this is set to epoch 35
4. The random 1000 Test_indices for chosen test points to compute influence is hard coded in the Jupyter notebook for model reproducibility. The indices for the 40000 training points is provided in training/training_indices.csv
5. The code to compute Influence functions is provided for reproducibility, but to avoid computation, a direct CSV with the required influences is provided directly in training/outdir/influences.csv. To use them, begin running code from the Visualising Influence Results section, (ensure that everything up to the Training section has been run!)