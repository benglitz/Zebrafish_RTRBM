# Zebrafish RTRBM #

This code base contains all model implementations and Jupyter notebooks accompanying:

**The Recurrent Temporal Restricted Boltzmann Machine Captures Neural Assembly Dynamics in Whole-brain Activity** <br/>
*Sebastian Quiroz Monnens, Casper Peters, Kasper Smeets, Luuk Willem Hesselink, Bernhard Englitz* <br/>
bioRxiv 2024.02.02.578570; doi: https://doi.org/10.1101/2024.02.02.578570

## Installation ##

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python
3. Open an anaconda prompt and create a new environment with: `conda create -n zebrafish_rtrbm python=3.9`.
4. Activate the created conda environment by running: `conda activate zebrafish_rtrbm`
5. Clone the contents of this repository to a desired directory
6. Navigate to the cloned repository directory and run: `python setup.py develop`
7. Install jupyter notebook by running: `conda install jupyter`
8. Install Pytorch by following install instructions at https://pytorch.org/get-started/locally/

## Data availability ##

All accompanying data necessary to replicate results is hosted at https://osf.io/tx2vz/?view_only=94cbae609ad64fb58340728ede721f89. Please note that, for storage-saving reasons, the RBM and RTRBM files for figure 3 have been converted to zip files, these files need to be unzipped before they can be used. Additionally, please keep the same file structure as found in the data repository in order to run the notebooks.
