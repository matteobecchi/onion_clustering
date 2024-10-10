# tropea_clustering
tropea-clustering (the newest version of onion-clustering) is a Python package for single-point time-series clustering. 

Author: Matteo Becchi

## Developement history
This version of onion clustering is meant to be used as an external library, and complies with the scikit-learn format. If you are looking for the standalone onion clustering version, you can find it at https://github.com/matteobecchi/timeseries_analysis. However, be aware that the standalone version has been last updated on September, 2024 and is no longer supported or mantained. We reccomand using this version. 

## Installation
To get `tropea-clustering`, you can install it with pip

``pip install tropea-clustering``

The `examples/` folder contains examples of usage. 

## Overview
Onion Clustering is an algorithm for single-point clustering of time-series data. It performs the clustering analyses at a specific time-resolution $\Delta t$, which is the minimum lifetime required for a cluster to be characterized as a stable environment. The clustering proceeds in an iterative way. At each iteration, the maximum of the cumulative distribution of data points is identified as a Gaussian state (meaning, a state characterized by the mean value and the variance of the signal inside it). The time-series signals are sliced in consecutive windows of duration $\Delta t$, and the windows close enough to the state's mean are classified as belonging to that state. These signals are then removed from the analysis, in order to enhance the resolution on the still unclassified signals at the next iteration. At the end of the process each signal windows is thus either classified in one of the identified states, or labelled as "unclassified" at that specific time resolution. 

Performing this analysis at different values of the time resolution $\Delta t$ allows to automatically identify the optimal choice of $\Delta t$ that maximizes the number of environments correctly separated, and minimizes the fraction of unclassified points. Complete details can be found at https://doi.org/10.1073/pnas.2403771121.

## Dependencies
- [numpy](https://numpy.org)
- [scipy](https://docs.scipy.org/doc/scipy/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)

For plotting the results, you will need also 
- [matplotlib](https://matplotlib.org)
- [plotly](https://plotly.com/graphing-libraries/) (optional)
- [kaleido](https://pypi.org/project/kaleido/) (optional)

## How to cite us
If you use tropea-clustering (or onion-clustering) in your work, please cite https://doi.org/10.1073/pnas.2403771121. 

## Aknowledgements
We developed this code when working in the Pavan group, https://www.gmpavanlab.com/. Thanks to Andrew Tarzia for all the help with the code formatting and documentation, and to Domiziano Doria, Chiara Lionello and Simone Martino for the beta-testing. 

The work was funded by the European Union and ERC under projects DYNAPOL and the NextGenerationEU project, CAGEX. 
