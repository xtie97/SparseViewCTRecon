# Sparse-View CT Reconstruction

**This repository contains code for reconstructing CT images from sparse-view projection data. The code is written in Python and uses the PyTorch and MONAI libraries.** Check our [**final report and slides**](https://github.com/xtie97/SparseViewCTRecon/tree/main) and [**webpage**](https://sites.google.com/wisc.edu/cs766-spring-2023-sparseview/home)📜.

## Usage:
Run the reconstruction script:
```
python train_grad.py -c config/hyperparam_grad.json
```
or 
```
python train_sino.py -c config/hyperparam_sino.json
```
To evaluate the model: 
```
python evaluate_grad.py -c config/hyperparam_grad.json
```
or 
```
python evaluate_sino.py -c config/hyperparam_sino.json
```
