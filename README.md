# Tumor Predictiors

This repo contains scripts for our DL final project models. We are estimating whether a given ovarian cancer is recurrent or not using clinical and epigenomic. By training on these definitively labelled cases of where the tumor has been labelled as recurrent or not by trained medical staff, the hope is that our model can then prospectively look at patient data prospectively, in the future, to predict whether they are likely to have a recurrent ovarian tumor at the point of care, using epigenomic and point-of-care clinical history features.



## Environment Setup

We recommend using `conda` to setup the required python environment for the project.

```bash
conda create -n dope python=3.7
conda install -c anaconda jupyter
conda install -c anaconda pandas
conda install tensorflow
conda install -c conda-forge xgboost
conda install -c anaconda numpy
conda install -c anaconda scikit-learn
conda install -c anaconda matplotlib
```

As the dataset provided will exceed the limitation of Github file size, we will use [Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Please follow the instructions to install it so everything can run successully.


## Data
TCGA Ovarian Cancer 
 
