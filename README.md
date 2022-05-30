# Testbed Implementation of FedBalancer on Flower

This repository contains the testbed implementation and corresponding experiments of the paper:

> [MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
> 
> Jaemin Shin et al., [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients](https://arxiv.org/abs/2201.01601)

## System Requirements

This system is written and evaluated based on the following setup:
- Server: A ```Ubuntu 18.04``` server 
- Clients: 21 Android smartphones as illustrated in ```Table 4``` of our paper, with Android OS version $\geq$ 7.0

As an alternative setup, you can use general Ubuntu servers with other Android smartphones.

## Installation - Server

We recommend you to setup Python environment using ```pyenv-virtualenv``` based on ```Developer Machine Setup``` in this [link](https://flower.dev/docs/getting-started-for-contributors.html) as follows:

## How to run

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
