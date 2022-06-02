# Testbed Implementation of FedBalancer on Flower

This repository contains the testbed implementation and corresponding experiments of the paper:

> [MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
> 
> Jaemin Shin et al., [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients](https://arxiv.org/abs/2201.01601)

## System Requirements

This system is written and evaluated based on the following setup:
- Server: A ```Ubuntu 18.04``` server running ```Python 3.7.12``` with ```pyenv-virtualenv```
- Clients: 21 Android smartphones as illustrated in ```Table 4``` of our paper, with Android OS version $\geq$ 7.0

As an alternative setup, you can use general Ubuntu servers with other Android smartphones.

## Installation - Server

- Install Python build dependencies
```
$ sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```
- Install [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv). We recommend using [pyenv-installer](https://github.com/pyenv/pyenv-installer) for Ubuntu servers.
- Setup Flower and install required packages in development mode (as pip install -e)
```
$ pyenv install 3.7.12
$ git clone https://github.com/jaemin-shin/flower-FedBalancer-testbed.git
$ cd flower-FedBalancer-testbed
$ ./dev/venv-create.sh
$ ./dev/bootstrap.sh
```

<!-- We recommend you to setup Python environment using ```pyenv-virtualenv``` based on ```Developer Machine Setup``` in this [link](https://flower.dev/docs/getting-started-for-contributors.html) as follows: -->

## Dataset setup

We evaluated based on the [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
- Go to ```data/``` and follow instructions on preprocessing the benchmark dataset
- Make ```data``` directory in the ```asset``` directory of Android client in ```android/client```
- Copy all the contents from the ```data/har``` directory to ```data``` of asset directory

## TensorFlow Lite model setup for on-device training

- Go to ```android/tflite_convertor``` and run ```python convert_to_tflite.py```
- Make ```model``` directory in the ```asset``` directory of Android client in ```android/client```
- Copy all the contents from the ```android/tflite_convertor/tflite_model``` directory to ```model``` of asset directory

## How to run 

- Prepare 21 Android devices and install the Android client app in ```android/client```
TBD

## Note

This repository is built top on a forked repository of [Flower](https://github.com/adap/flower), which is a friendly federated learning framework that is maintained by a wonderful community of researchers and engineers. Therefore, this repository contains same license file as in Flower. 

## Citation

If you publish work that uses this repository, please cite FedBalancer and Flower as follows:

(FedBalancer bibtext - TBD)

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```
