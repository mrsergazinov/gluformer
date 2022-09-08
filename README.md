# Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification
This is the official implementation of the paper "Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification" ([link]()).

---

Table of Contents:
- [1. Overview](#1-overview)
- [2. How to work with the repository?](#2-how-to-work-with-the-repository)
  - [2.1 Setting up the environment](#21-setting-up-the-environment)
  - [2.2 Running the experiment](#22-running-the-experiment)
  - [2.3 Running the scripts](#23-running-the-scripts)
- [3. How to cite](#3-how-to-cite)

---

## 1. Overview
The code is organized as follows:
- `cache`:
  - `visualize_*` contains code for reproducing plots from the paper.
- `gludata` 
  - `data` folder containing the data.
  - `data_loader.py` provides the implementation of PyTorch `Dataset` for the data.
- `gluformer` provides our model implementation.
- `trials` contains outputs from the experiments on the real / synthetic data sets.
  - `trials.txt` provides commands to run the experiments.
- `utils` contains common tools for model training / evaluation.
- `experiment.ipynb` provides an example on the synthetic data of how to train and evaluate our model.
- `model_*` scripts for model training and evaluation respectively that can be run from the command line. 

Additionally, we provide `environment.yaml` that gives a snpshot of our `conda` build for reproducibility. 

## 2. How to work with the repository?
The repository provides a self-contained code to repoduce the results from the paper on the glucose and synthetic data sets. Below, we outline some futher instructions.

### 2.1 Setting up the enviroment
We recommend use `conda` to create a virtual environment for this project. Once you have pulled the code and installed `conda` on your system, you need to run the following command from the root (repository) folder: `conda env create -f environment.yml`. Once the necessary packages are installed, you can activate your environment by running `conda activate gluformer`.

### 2.2 Running the experiment
We suggest to start the explortion of our model by running the model on the synthetic data. The code for this is provided in the `experiment.ipynb` notebook. The code is largely self-contained as it gives the implementation of both the training and the evaluation loops. Additionally, the notebook contains the data generating function for the synthetic data. All of this is done to expose the use to the inner workings of the model and ease potential extensions to the other data sets. 

### 2.3 Running the scripts
We provide `model_train.py` and `model_eval.py` scripts that can be run from the command line and give an implementation of the training and evaluation loops respectively. Both scripts expect to be run from the root (repository) folder and have all dependencies (specified in the `environment.yaml` file) installed. For an example of what parameters each script takes, see the `trials.txt` file in the `trials/` folder.

## 3. How to cite
```
@inproceedings{sergazinov2022gluformer,
    title={Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification},
    author={Renat Sergazinov and Mohammadreza Armandpour and Irina Gaynanova},
    booktitle={{arXiv}},
    year={2022},
}
```