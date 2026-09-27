# TDCL

This repository contains the code for the software simulations and related figures for "Trace Differences Enable Unsupervised Contrastive Local Online Learning for Efficient Streaming Data Processing".

## Installation
In a virtual environment, run
```shell
pip install -r requirements.txt
```

## Experiments
All experiments can be run by calling scripts from `root`.
These train a network of fully connected LIF neurons with TDCL and after training probe the network with a linear classifier.
Results can be found under `results/`.
Data is downloaded automatically, except for the `WEAR` dataset, which can be found here: https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/right_w/

### NMNIST
```shell
PYTHONPATH=. python tdcl/train_tdcl_ff_full_online_nmnist.py
```

### SHD single layer
For training a single layer on different kernel configurations
```shell
PYTHONPATH=. python tdcl/train_tdcl_ff_full_online_shd.py
```
with the following flags controlling the kernel:
- `-gf`: gain fast
- `-gs`: gain slow
- `-bf`: beta fast
- `-bs`: beta slow

### SHD 6 layers
```shell
PYTHONPATH=. python tdcl/train_tdcl_ff_full_online_shd_6_layer.py
```
This probes the network after training at every layer

### SHD early training
Disclaimer: this records data of a few GB.
```shell
PYTHONPATH=. python tdcl/record_early_training_dynamics.py
```
It allows for the same kernel control via the same flags as SHD single layer runs

### WEAR
```shell
PYTHONPATH=. python tdcl/train_tdcl_ff__full_online_wear.py
```
This requires the `csv` files under `data/WEAR/right_w/`.