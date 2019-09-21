# MLCNN for Multivariate Time Series Forecasting

We implement the Multi-Level Construal Neural Network (MLCNN) model to perform multivariate time series forecasting.

### Usage

Please install [Git Large File Storage](https://git-lfs.github.com/) first and then use 

```shell
git lfs clone https://github.com/smallGum/MLCNN-Multivariate-Time-Series.git
```

to download all the datasets and codes.

You can find the dataset in the `data/` folder.

Examples with parameter grid search to run different datasets are in `runTraffic.py`, `runEnergy.py` and `runNASDAQ.py`.

### Environment

Python 3.6.7 and Pytorch 1.0.0

### Acknowledgements

This multivariate time series forecasting framework was implemented based on the following two repositories:

+ [LSTNet](https://github.com/laiguokun/LSTNet)
+ [SOCNN](https://github.com/mbinkowski/nntimeseries)

Some codes and design patterns are borrowed from these two excellent frameworks.