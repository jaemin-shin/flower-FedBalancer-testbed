# Testbed Implementation of FedBalancer on Flower

This repository contains the testbed implementation and corresponding experiments of the paper:

> [ACM MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
> 
> Jaemin Shin et al., [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients](https://arxiv.org/abs/2201.01601)

## System Requirements

This system is written and evaluated based on the following setup:
- Server: A ```Ubuntu 18.04``` server running ```Python 3.7.12``` with ```pyenv-virtualenv```
- Clients: 21 Android smartphones as illustrated in ```Table 4``` of our paper, with Android OS version $\geq$ 7.0

As an alternative setup, you can use general Ubuntu servers with other Android smartphones.

## How to run: Step 1 - Server setup

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

## How to run: Step 2 - Dataset setup

We evaluated based on the [UCI-HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
- Go to ```data/``` and follow instructions on preprocessing the benchmark dataset
- Make ```data``` directory in the ```asset``` directory of Android client in ```android/client```
- Copy all the contents from the ```data/har``` directory to ```data``` of asset directory

## How to run: Step 3 - TensorFlow Lite model setup for on-device training

- Go to ```android/tflite_convertor``` and run ```python convert_to_tflite.py```
- Make ```model``` directory in the ```asset``` directory of Android client in ```android/client```
- Copy all the contents from the ```android/tflite_convertor/tflite_model``` directory to ```model``` of asset directory

## How to run: Step 4 - Perform latency sampling on 21 Android devices
<img src="https://github.com/jaemin-shin/flower-FedBalancer-testbed/blob/main/android_app_img.jpg" width="200" />

Before running federated learning, we recommend you to perform latency sampling on participating devices.
- Prepare 21 Android devices and install the Android client app in ```android/client```
- On Android client app, enter client ID, server machine IP, and port in the app screen on 21 devices.
  - Each devices should have different client ID between 1 - 21.
  - Here, you should enter the port that you will run federated learning. We recommend you to use 8999 port.
- On server, run following below:
  - For efficient latency sampling, this runs 21 server instances with different ports. We configured each instance to use ```8999 - client_id``` port (e.g., server with port 8998 is used by a client with ID 1).
```
$ cd android
$ ./latency_sampling_run.sh
```
- On Android client app, press 'LOAD' button to load client data on each of 21 devices.
- On Android client app, press 'SETUP FOR LATENCY SAMPLING' button to prepare the app for latency sampling on 21 devices.
- On Android client app, press 'START' to start latency sampling on 21 devices.

After sampling round completion time for 10 rounds on each device, the server instances terminate.
Sampled latency information of each client are saved in ```android/log/client_x_latency.log```.

## How to run: Step 5 - Run federated learning!

- Terminate and re-run the Android client app on 21 Android devices.
- On Android client app, enter client ID, server machine IP, and port in the app screen on 21 devices.
  - Each devices should have different client ID between 1 - 21.
  - Here, enter the port that you will run federated learning. We recommend you to use 8999 port.
- On server, run following below:
  - We prepared the config files in ```configs``` so that you could run to test FedBalancer and baselines in our paper.
  - For each parameter item in a config file, please refer to [FedBalancer](https://github.com/jaemin-shin/FedBalancer) for explanation.
```
$ cd android
$ python server.py --config=configs/{config_name}.cfg # e.g. python server.py --config=configs/fedavg_ddl_1T.cfg
```
- On Android client app, press 'LOAD' button to load client data on each of 21 devices.
- On Android client app, press 'SETUP' button to prepare the app for federated learning on 21 devices.
- On Android client app, press 'START' to start latency sampling on 21 devices.

## Parsing Results

- After running federated learning, the result log is saved in ```android/log```.
- Please refer to the jupyter notebook ipynb scripts in ```results_parsing``` directory.

## Note

In this repository, we "simulate" deadlines for each training round. As Flower uses [Futures](https://docs.python.org/3/library/asyncio-future.html) Python library to implement client training at a round, it cannot terminate a round while waiting for a client to finish their task. Thus, we implemented our server to wait until every clients to finish their task but to reject client updates that arrived later than the round deadline. For this reason, we measure ```elapsed_time``` variable in ```src/py/flwr/server/server.py``` to measure the time elapsed during FL assuming that the round terminates after the deadline.

## Acknowledgement on Flower

This repository is built top on a forked repository of [Flower](https://github.com/adap/flower), which is a friendly federated learning framework that is maintained by a wonderful community of researchers and engineers. Therefore, this repository contains same license file as in Flower. 

## Citation

If you publish work that uses this repository, please cite FedBalancer and Flower as follows:

```bibtex
@inproceedings{10.1145/3498361.3538917,
author = {Shin, Jaemin and Li, Yuanchun and Liu, Yunxin and Lee, Sung-Ju},
title = {FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients},
year = {2022},
isbn = {9781450391856},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3498361.3538917},
doi = {10.1145/3498361.3538917},
booktitle = {Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services},
pages = {436–449},
numpages = {14},
location = {Portland, Oregon},
series = {MobiSys '22}
}
```

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```
