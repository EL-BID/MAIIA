Miscellaneous
=============

Here are some miscellaneous tips and potentially useful ideas for accessing extra 
data etc.

Access Sentinel Data
--------------------

Sign up for a free copernicus account and make a note of your username and passsword at `copernicus scihub`_.

.. _copernicus scihub: https://scihub.copernicus.eu/dhus/#/self-registration

Install `SentinelSat`_ on PyPI::

    pip install sentinelsat

.. _SentinelSat: https://sentinelsat.readthedocs.io/en/stable/

Now in python, create an API session like::

    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

    api = SentinelAPI('username', 'password', api_url='https://scihub.copernicus.eu/dhus')


Hardware requirements
--------------------

To select the machine for a machine learning task is challenging because you have to consider many factors such as portability, processing speed, and the graphics processing capability among others. 
We distiguish the case where the user has access to limit settings and the case where the user has a considerable computing power.

Personal work-station
^^^^^^^^^^^^^^^^^^^^^
GPU

An NVIDIA GPU is preferable because of the available frameworks and APIs (CUDA and CuDNN) compatible with major deep learning frameworks such as TensorFlow and PyTorch. The latest generations of NVIDIA GPUs such as the GeForce RTX based on Turing architecture are AI-enabled with Tensor cores which makes them suitable for deep learning.

RAM

Although a minimum of 8GB RAM can do the job, 16GB RAM and above is recommended for most deep learning tasks.

CPU

When it comes to CPU, a minimum of 7th generation (Intel Core i7 processor) is recommended. However, getting Intel Core i5 with Turbo Boosts can do the trick. If one opts for a desktop then selecting the right combination of CPU and motherboard that match your GPU specifications is recommended. In that case, the choice of the number of PCIe lanes ( PCIe lanes determine the speed of transferring data from CPU RAM to GPU RAM) should also be taken into consideration (4-16 PCIe lanes is best for most deep learning tasks).

Storage

Storage is also an important factor, specifically due to the increasing size of deep learning datasets requiring higher storage capacity. For example, Imagenet, one of the most popular datasets for deep learning, is 150 GB in size and consists of more than 14 million images across 20,000 categories. Although SSD is recommended for its speed and efficiency, you can get an HDD at a relatively cheaper price to do the job. However, if you value speed, price and efficiency then a hybrid of the two is the best option.


In summary, if you are going to work on low-computation machine learning tasks that can be easily handled through complex sequential processing then you don’t need a GPU. For such tasks, a laptop with a minimum of 8GB ram, 500HDD and turbo boost core i5 Intel processor will do fine.
If you intend to work on slightly computationally-intensive deep learning tasks and large datasets, then it is advisable that you consider a GPU. There are two options to this: (1) you can buy a powerful laptop with GPU if portability is critical; (2) If portability is not an issue then you can set up a desktop and connect it with your laptop for remote access. For such tasks both old and new Nvidia GPUs such as Nvidia NVS 310, GT, GTS, and RTS with a minimum of 2GB VRAM, 16-64GB RAM are recommended.


Cloud computing
^^^^^^^^^^^^^^^

If you are a firm regularly working on complex deep learning problems then it is advisable to invest in cloud services like Azure, AWS and Google Cloud. In this case, it's advisable to use a laptop for preprocessing and debugging, and train on the cloud where GPU instances now go for as low as $0.7/hour on AWS. Cloud providers have several GPU offerings. For EC2 instance type, the AWS’s general-purpose GPU instances offers Nvidia’s V100 GPU, which can deliver over 100 TFLOPS peak performance for training and inference. 
There are many Amazon EC2 GPU instances options, some with the same GPU type but different CPU, storage and networking options. For example, P3 instances provide access to above=mentioned NVIDIA V100 GPUs based on NVIDIA Volta architecture and you can launch a single GPU per instance or multiple GPUs per instance (4 GPUs, 8 GPUs). A single GPU instance p3.2xlarge can be your daily driver for deep learning training. And the most capable instance p3dn.24xlarge gives you access to 8 x V100 with 32 GB GPU memory, 96 vCPUs, 100 Gbps networking throughput ideal for distributed training.

You can access and work on this cloud infrastructure from any low hardware setting preferably running on Linux Operating system (Window works as well). Thus, 8GB RAM, 80GB HDD with an Intel Gold 6130 CPU and a reliable network  connectivity would be enough.

