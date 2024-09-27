.. onion-clustering documentation master file. 
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   :caption: onion-clustering
   :maxdepth: 2

   onion-clustering <modules>
   Gaussian states <classes>

============
Introduction
============

| GitHub: https://github.com/matteobecchi/onion_clustering


:mod:`.onion-clustering` is a Python package for single-point time-series clustering.

Development history
-------------------
This version of onion clustering is meant to be used as an external library, and complies with the scikit-learn format. If you are looking for the standalone onion clustering version, you can find it at https://github.com/matteobecchi/timeseries_analysis. However, be aware that the standalone version has been last updated on September, 2024 and is no longer supported or mantained. We reccomand using this version. 

Installation
------------

To get :mod:`.onion-clustering`, you can install it with pip::

  $ pip install onion-clustering

The :mod:`examples/` folder contains an example of usage. From this folder, download the example files as reported in the script :mod:`example_script_uni.py` and then run it with::

  $ python3 example_script_uni.py 

Overview
--------

To be written.

Dependencies
------------
- numpy (https://numpy.org)
- scipy (https://docs.scipy.org/doc/scipy/index.html)
- scikit-learn (https://scikit-learn.org/stable/)

Acknowledgements
----------------

We developed this code when working in the Pavan group, https://www.gmpavanlab.com/. Thanks to Andrew Tarzia for all the help with the code formatting and documentation, and to Domiziano Doria, Chiara Lionello and Simone Martino for the beta-testing. 

The work was funded by the European Union and ERC under projects DYNAPOL and the NextGenerationEU project, CAGEX. 


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
