
Getting Started
===============

It's recommended to work in a `Tmux`_ window so that you can run tasks (like training 
models or running inference) in the background and detach from the session.

Very briefly, to create a new Tmux session::

    tmux new -s your_session_name

To exit a session, type **ctrl+b** then **d**.

To attach to a session::

    tmux a -t your_session_name

To kill a session (while attached), type **ctrl+b** then **x**.

.. _Tmux: https://linuxize.com/post/getting-started-with-tmux/

Before running anything, make sure you're in the docker container environment and have 
the necessary volumes mounted as described in :doc:`install` which are linked to the 
directories specified for finding models and datasets in your :ref:`configuration`::

    $ cd gim-cv
    $ docker-compose up -d `# if not running already`
    $ docker exec -it gim_cv_container /bin/bash
    $ conda activate gim_cv_gpu

.. _configuration:

Configuration
-------------

Configuration is handled by editing the ``config.yml`` file in the root directory of 
the source repository. This is mostly used to provide convenient shortcuts for paths. 
Variables defined therein are then accessible via the :mod:`gim_cv.config` module 
which can be used, for example, as follows::

    import gim_cv.config as cfg
    print(cfg.models_path)

The contents of the ``cfg`` module will then reflect the contents of ``config.yml`` 
at the time it was imported.

Certain paths such as ``cfg.training_data_path`` and ``cfg.models_path`` should be 
set to match the directories where these are stored. For example, if you have mapped 
a disk containing saved models into a subdirectory of
``/home/root/saved_models/`` then you should set the ``models_path`` 
entry in ``config.yml`` as::

    project_dir     : &PROJECT_DIR /home/root
    models_path : !join [*PROJECT_DIR, /saved_models]

Note the ``&TAG`` and ``*TAG`` directives can be used to reference previously defined 
configuration variables. This way, later on you can do something like::

    subdirectory = 'fancy_segmentation_architecture_v2'
    weights_name = 'best_network_weights.hdf5'
    model.save_weights(cfg.models_path / Path(f'{subdirectory}/{model_name}'))

Similarly when :doc:`datasets` are defined in :mod:`gim_cv.datasets` these point 
to raster and vector files. Typically you will want to mount a datasets volume 
somewhere in ``/home/root/data/``. For example, typically I have a large disk for 
storing all training datasets and have this mounted at ``/home/root/data/volumes/datasets``.
``config.yml`` then contains the entries::

    data_path       : &DATA_PATH !join [*PROJECT_DIR, /data]
    volumes_data_path : &VOLUMES_DATA_PATH !join [*DATA_PATH, /volumes]
    training_data_path : &TRAINING_DATA_PATH !join [*VOLUMES_DATA_PATH, /datasets]

Now in the datasets module, you can define something like::

    my_dataset = Dataset(tag='leuven_orthophoto_2018_summer',
                         image_paths=[cfg.training_data_path / Path('leuven/ortho_2018.tif')],
                         mask_paths=[cfg.training_data_path / Path('leuven/ground_truth_2018.shp')],
                         spatial_resolution=0.25)

General Use
-----------

Once the codebase is installed and configured, you have a few options for how to use it.

If you are a user rather than a developer, you will probably want to use the 
functionality provided in the existing scripts for :doc:`training` and 
:doc:`inference` outlined in those sections. You may also want to use the 
library code in a jupyter notebook session to do some custom analysis using 
segmentation results. It's probably also understanding how the :doc:`datasets` 
module is used. These will be explained in the following pages.

Using Jupyter Notebooks
^^^^^^^^^^^^^^^^^^^^^^^

You can use Jupyter notebook by launching it in the container, then opening
your browser at port 8888 of localhost on the host OS::

    $ jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root

This is aliased in the docker environment as simply::

    $ jn

You should open the *second* link which appears in::

    [I 12:12:17.801 NotebookApp] Serving notebooks from local directory: /home/root
    [I 12:12:17.801 NotebookApp] The Jupyter Notebook is running at:
    [I 12:12:17.801 NotebookApp] http://6d663593d11f:8888/?token=82d0ab41090cc42f5793528225b46f324e8cbc02d903cb85
    [I 12:12:17.801 NotebookApp]  or http://127.0.0.1:8888/?token=82d0ab41090cc42f5793528225b46f324e8cbc02d903cb85


i.e. that which begins with http://127.0.0.1:8888/

When working locally this link will work immediately (since port 8888 is mapped to 8888 on the host OS
in ``docker-compose.yml``).

When working on a remote machine, you will need to first open an SSH connection to the remote machine on port 8888
and forward it to a local port (say 9999) before opening the link (edited so that 8888 -> 9999) in your 
browser. Run::

    $ ssh -NfL 9999:localhost:8888 cloud_machine

Where it is assumed that you have configured an entry named cloud_machine in your ``~/.ssh/config`` file 
and provided the appropriate private key file therein. It is recommended to set an alias for the notebook 
SSH connection in your ``~/.bashrc`` or ``~/.zshrc`` file so you don't have to type it every time.

Notes
-----

Sometimes you can encounter an error with file permissions when files are created in the docker container 
and you try to change or delete them from outside it. If this is happening in ``directory``, the quick 
fix is just to run::

    sudo chown -R youruser:youruser directory

Where ``youruser`` should be replaced with whatever your user account is (``root`` etc.).

Some ideas for more permanent fixes are `here <https://vsupalov.com/docker-shared-permissions/>`_.

Development
-----------

Developing with the source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are a developer and extending the functionality of the codebase, there are two 
common scenarios you will encounter in which you will want to edit code.

In the first one, you are working locally on a machine with a GUI text editor. In this 
case, since the code repository is mirrored between docker and the host OS, you can 
simply use your text editor or IDE of choice to edit the repository on the host OS.

In the second one, you are working through SSH on a cloud machine. In this case, 
you can launch a Jupyter notebook server in the container 
connect to it on your local machine and use the jupyter browser as a code editor.
A more sophisticated approach might be to use SSH remote-editing functionality such as 
exists in VSCode.

Testing
^^^^^^^

It is preferred that if you extend the codebase, you should write tests to make sure 
that you haven't broken anything. Currently the test coverage is incomplete but covers 
the most important aspects of the model training and inference pipelines.

Tests live in the ``tests`` directory of the project and are implemented using `pytest`_.
To run your tests, just execute ``pytest tests`` from the docker/conda environment in the 
root of the source repository.

.. _pytest: https://docs.pytest.org/en/stable/
