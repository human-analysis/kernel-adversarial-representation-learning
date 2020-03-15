# On the Global Optima of Kernelized Adversarial Representation Learning

By Bashir Sadeghi, Runyi Yu, and Vishnu Naresh Boddeti

## Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)

### Introduction

This code archive includes the Python implementation to impart fairness for
already trained representation or raw data by designing a kernelaized regressor as an encoder
for the new representation. The new representation aims to trade-off between utility
(i.e., the performance of target task) and the leakage of sensitive attribute (i.e. adversary performance).
Our closed-form solution is implemented for three different kernels, namely, Linear, Polynomial and Gaussian
and can be easily extended to any other kernel.
After learning the encoder, we freeze it and  train a real adversary and target networks.

### Citation

If you think this work is useful to your research, please cite:

    @inproceedings{sadeghi2019global,
      title={On the Global Optima of Kernelized Adversarial Representation Learning},
      author={Sadeghi, Bashir and Yu, Runyi and Boddeti, Vishnu},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={7971--7979},
      year={2019}
    }

**Link** to the paper: http://hal.cse.msu.edu/papers/kernel-adversarial-representation-learning/

### Setup
First, you need to download PyTorchNet by calling the following command:
> git clone --recursive https://github.com/human-analysis/kernel-adversarial-representation-learning.git

### Requirements

1. Require `Python3`
2. Require `PyTorch1.0`
3. Require `Visdom0.1.8.9`
4. Check `Requirements.txt` for detailed dependencies.

### Commands to Reproduce Results in Paper
## Synthetic Gaussian Dataset
~~~~
$ python3 -m visdom.server
$ python3 main.py --args args/args-gaussian.txt
~~~~

## Adult Dataset
~~~~
$ python3 -m visdom.server
$ python3 main.py --args args/args-adult.txt
~~~~

## German Dataset
~~~~
$ python3 -m visdom.server
$ python3 main.py --args args/args-german.txt
~~~~

## Yale-B
~~~~
$ python3 -m visdom.server
$ python3 main.py --args args/args-yaleb.txt
~~~~

## CIFAR-100
~~~~
$ python3 -m visdom.server
$ python3 main.py --args args/args-cifar.txt
~~~~

### General Usage Instructions
You need to run `main.py`.
#### Part A: Training the Encoder

1. Set the path to your input data and your dataset name for both training and test sets.
**Note:** Let the data created by `dataloader.py` contain three items, input data, target class label
and sensitive class label, respectively.
    Example in `args.txt`:
    ```
    dataset_train = Cifar100
    input_filename_train = ./train_input
    label_filename_train = ./train_label

    dataset_test = Cifar100
    input_filename_test = ./test_input
    label_filename_test = ./test_label
    ```

2. Set the dimentionality of your embedding `r` and data`ndim`, number of sensitive class label
    `nclasses_A`, and number of target class label `nclasses_T`.
    Example in `args.txt`:
    ```
    r = 19
    ndim = 64
    nclasses_A = 100
    nclasses_T = 20
    ```
3. Set the trade-off parameter (0<=`lambd`<=1) between privacy and utility.
**Note:** `lambd=0` is related to no privacy and `lambd=1` concerns totally
to hide the sensitive attribute.

4. Set the batch size required for training `batch_size_e` . Example in `args.txt`: `batch_size_e = 12500`.
**Note:** The ideal batch size is as large as the number of input samples.
 For the linear kernel the final encoder is the average between the encoders obtain for each batch.
 On the other hand, for the non-linear kernel, data is randomly sampled once.

5. Choose your kernel among three provided kernels: `Linear`, `Polynomial` and `Gaussian`.
**Note:** For Polynomial kernel (K = (x^T*y + c)^d), two hyper parameters are required to be set:
the constant part `c` and the exponent part `d` (`d` must be a natural number).
For Gaussian kernel (K = exp(-||x - y||^2 / sigma)), the variance  `sigma` should be set which is
a positive number.
    Example in `args.txt`:
    ```
   kernel = Polynomial
   c = 1
   d = 7

   #kernel = Gaussian
   #sigma = 80
   ```


#### Part B: Training the Real Adversary and Target Classifiers or Regressors

1. Visualization Settings.
The parameters for visdom to plot training and testing curves.

        1) the port number for visdom -- "port"
        2) the name for current environment -- "env"
        3) if you want to create a new environment every time you run the program or not -- "same_env".  If you do, set it "False"; otherwise, it's "True".

    Example in `args.txt`:
    ```
    port = 8093
    env = main
    same_env = True
    ```

2. Select the network for target and adversary and specify their task as a regression or classification.
Example in `args.txt`:
    ```
    model_type_A = Adversary
    model_type_T = Target
    loss_type_A = Classification
    loss_type_T = Classification
    evaluation_type_A = Classification
    evaluation_type_T = Classification
    ```

3. Finally, set the hyper parameters required to train and test the real adversary and target networks.
Example in `args.txt`:
    ```
    nepochs = 2000
    optim_method = Adam
    learning_rate_T = 5e-3
    learning_rate_A = 5e-3
    scheduler_method = MultiStepLR
    scheduler_options = {"milestones": [100, 200]}
    ```
