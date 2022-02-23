Installation
============

Overview
--------

The codebase takes the form of a python library, ``gim_cv``, embedded in a docker 
container together with various scripts (and jupyter notebooks) which cover common 
use cases such as training a segmentation model and running inference with a trained 
model to create segmented rasters.

Installation boils down to correctly configuring the ``docker-compose.yml`` file to 
perform any mapping of volumes to be used (where data and models live), then 
spinning up the docker container and entering the anaconda environment therein.

The source code on the host machine is mirrored into the container environment, so that 
you may edit the source in either (for example, in a text editor on your host machine, 
or when attached to a remote jupyter notebook session running in the container itself).

The codebase has been developed to run on a Linux OS, and possibly (hopefully!) on 
WSL v2 with nvidia/docker support.

Docker installation instructions
--------------------------------

Prerequisites are to install `docker`_ and `nvidia-docker`_ (to allow use of the GPU in 
the container). Follow the instructions at these links first and test that they work 
according to the documentation.

.. _docker: https://docs.docker.com/engine/install/ubuntu/#installation-methods
.. _nvidia-docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Setting NVIDIA runtime as default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure that your docker containers can access the GPU by default, edit/create the file 
``/etc/docker/daemon.json`` so that it looks like this::

    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }

If this is the first time you edit this file on your machine, restart the docker 
service now to make sure this takes effect::

    $ sudo systemctl restart docker.service

Linking disk volumes
^^^^^^^^^^^^^^^^^^^^

Before creating the docker container, first check the ``volumes`` entry of ``docker-compose.yml``.
If you intend to use/create disk-space-heavy resources (usually the case!) you will want to 
make sure you have the appropriate disk volumes linked to the container environment. Typically 
these will be volumes used for storing datasets and for storing models.

Suppose you have two disks (or EBS volumes).
Let's assume these correspond to ``models`` and ``datasets``. These might also be 
different directories of the same drive. You can look at the drive device paths by running 
``lsblk`` on the command line.

The default directories where datasets and models are looked for (according to ``config.yml``) are 
currently ``/gim-cv/data/volumes/datasets`` and ``/gim-cv/saved_models/ebs_trained_models``.

There are two different ways to mirroring the drives:

*    Mount the volumes at the appropriate locations within the project repository 
     before launching the container. Say the relevant drive partitions are at 
     ``/dev/nvme1n1p1`` for datasets and ``/dev/nvme2n1p1`` for models. 
     The binding will then be taken care of by the ``.:/home/root`` directive in ``docker-compose.yml``::

        $ sudo mount /dev/nvme1n1p1 /path/to/gim_cv/data/volumes/datasets
        $ sudo mount /dev/nvme2n1p1 /path/to/gim_cv/saved_models/ebs_trained_models

*    Mount the volume first on the host machine at, say, ``/mnt/bigdata/``. Now map the volumes 
     directly in ``docker-compose.yml``. You will want to add to the ``volumes`` entry 
     something like::

        - type: bind
          source: /mnt/models
          target: /home/root/saved_models/ebs_trained_models
        - type: bind
          source: /mnt/datasets
          target: /home/root/data/volumes/datasets

These default directories can be changed as you see fit through :ref:`configuration`.

Once the container is running (see the following section) you should check that the files 
you are expecting to be there are appearing where you have mapped them. For example::

    $ docker exec -ti gim_cv_container /bin/bash
    $ ls /home/root/saved_models/bigdata_models

If these are not present, you have done something wrong.

*Bear in mind when working across multiple machines that these data drives may follow a different 
structure. It makes sense then to add ``docker-compose.yml`` to ``.gitignore`` and maintain separate 
versions (differing in the ``volumes`` section) for each machine.*


Build docker image containing the code and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now use ``docker-compose`` to build the image specified in ``Dockerfile``. This creates an
image called ``gim_cv``, installs geospatial libraries like GDAL on the OS image, installs 
Anaconda, copies the source code of this repository into the image and finally builds the 
python enviornment.

From the root of the ``gim_cv`` repository, run::

    docker-compose up -d


If you run into an error like ``Couldn't connect to docker daemon``, do the
following and try again::

    sudo systemctl unmask docker
    sudo systemctl start docker


Create a container from the image and launch it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've run ``docker-compose up -d``, a specific instance of this image
(a container) called ``gim_cv_container`` is created, with the specifics configured 
as in ``docker-compose.yml`` (port-forwarding, volume mirroring with the host OS etc.).

Check that your container is up and running by running ``docker ps``. You should see 
something like this::

    CONTAINER ID        IMAGE               COMMAND               CREATED             STATUS              PORTS                    NAMES
    44315329363a        gim_cv        "tail -f /dev/null"   29 seconds ago      Up 19 seconds       0.0.0.0:8888->8888/tcp   gim_cv_container


Optional - disable firewall (in case of problems on CentOS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If, on Linux, you run into the issue that the containers have no internet
(eg apt update fails), do::

    $ sudo systemctl disable firewalld
    $ systemctl stop docker
    $ systemctl start docker


Attach to the container
^^^^^^^^^^^^^^^^^^^^^^^

Now attach to the container to run code in the environment. It's recommended to do this
in a `tmux`_ session so that you can run things in the background by detaching 
if you desire::

    $ docker exec -it gim_cv_container /bin/bash

.. _tmux : https://linuxize.com/post/getting-started-with-tmux/

Finally enter the python environment::

    $ conda activate gim_cv_gpu

and install ``gim_cv`` as a local python package in editable mode with::

    $ pip install -e .

Press ctrl+D or type 'exit' to exit.

Note that if you shut down the container with ``docker-compose down``, you will have 
to repeat this last step of installing the local package the next time you recreate it.

Testing your installation
-------------------------

Here are some quick checks you can do to see that things are working.

    * Check that the GPU is accessible from the python interpreter::

        import tensorflow as tf
        assert tf.test.is_gpu_available()

    * Import the gim_cv module::

        import gim_cv

    * Check that any mountpoints contain the expected files (after you have done
      :ref:`configuration`)::

        from pathlib import Path
        import gim_cv.config as cfg

        print(list(Path(cfg.models_path).glob('*')))

    * Run pytest and check that the tests pass. In ``/home/root/``, run::

        $ pytest tests

Notes
-----

The ``docker-compose.yml`` file also creates a `Splash`_ container which 
runs a splash server in the background. This is useful for webscraping 
interactive websites (as is done in :py:mod:`gim_cv.scrapers.vl_orthos`
for obtaining Flemish orthophoto download links). This has not been used 
in a while and most likely you will not need it.

.. _Splash: https://splash.readthedocs.io/en/stable/api.html
