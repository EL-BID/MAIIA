.. AI Framework documentation master file, created by
   sphinx-quickstart on Thu Apr  8 12:04:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GIMs Deep Learning pipeline for informal settlement detection!
=====================================================

This documentation provides an overview of the functionality of the GIM's Computer Vision (gim-cv) package for informal settlement delineation.

The primary function of this package is to provide simple interfaces through which deep-learning-based computer vision techniques can be used to learn from and make inferences on georeferenced raster data with a view of the delineation of informal settlements.


The next sections provide installation, basic usage instructions and examples of it's segmentation tool.

.. toctree::
   :maxdepth: 2
   :caption: Segmentation models:

   install
   getting_started
   datasets
   preprocessing
   models
   training
   inference
   polygonisation
   misc

Detailed API documentation
--------------------------

Detailed API documentation for the functions and classes defined in each of the 
modules herein can be found in the following sections and is intended for developers.

.. toctree::
   :maxdepth: 2
   :caption: Main package:

   gim_cv

The segmentation model API documentation can be found here:

.. toctree::
   :maxdepth: 2
   :caption: Segmentation models:

   gim_cv.models


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
