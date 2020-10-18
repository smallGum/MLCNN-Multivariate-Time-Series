# MLCNN for Multivariate Time Series Forecasting

This repository provides the code for the paper [Towards Better Forecasting by Fusing Near and Distant Future Visions](https://arxiv.org/abs/1912.05122), accepted by AAAI 2020.

### Usage

You can find the `Energy` and `NASDAQ` dataset in the `data/` folder. As For `Traffic` dataset, you can find it in [LSTNet data repository](https://github.com/laiguokun/multivariate-time-series-data/tree/master/traffic).

Examples with parameter grid search to run different datasets are in `runTraffic.py`, `runEnergy.py` and `runNASDAQ.py`.

### Environment

Python 3.6.7 and Pytorch 1.0.0

### Acknowledgements

This multivariate time series forecasting framework was implemented based on the following two repositories:

+ [LSTNet](https://github.com/laiguokun/LSTNet)
+ [SOCNN](https://github.com/mbinkowski/nntimeseries)

Some codes and design patterns are borrowed from these two excellent frameworks.