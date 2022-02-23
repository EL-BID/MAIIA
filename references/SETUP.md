Installation and setup instructions
=============================================

This report helps go throughnthe installation and setup steps to deploy the current framework.
It is recommended to make the deployment on a clean AWS EC2 instance. The test was performed by creating a new instance with the following settings:
* AMI: Deep Learning AMI (Ubuntu 18.04) Version 43.0 - ami-0f521020beb162fd3
* Instance TYPE: gd4n.2xlarge
* Configure Instance Details - Subnet: eu-west-2a
* Storage: 299 GB


## Docker installation instructions

Prerequisites are to install [docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (to allow use of the GPU in 
the container), following the instructions at these links first and test that they work 
according to the documentation.

### Setting NVIDIA runtime as default

To ensure that your docker containers can access the GPU by default, edit/create the file 
``/etc/docker/daemon.json`` so that it looks like this::
```bash
    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
```

If this is the first time you edit this file on your machine, restart the docker 
service now to make sure this takes effect:
```bash
$ sudo systemctl restart docker.service
```

### Build docker image containing the code and dependencies

Use docker-compose to build the image specified in `Dockerfile`. This creates an
image called `gim_cv` on Debian, installs Anaconda, copies the source
code of this repository into the image before finally installing python
dependencies.

```bash
$ sudo docker-compose up -d
```

If you run into an alert like `Command *docker-compose* not found`, do the
following and try again:

```bash
$ sudo apt install docker-compose
```


If you run into an error like `Couldn't connect to docker daemon`, do the
following and try again:

```
sudo systemctl unmask docker
sudo systemctl start docker
```


### Create a container from the image and launch it

Once you've run `docker-compose up -d`, a specific instance of this image
(a container) called
`gim_cv_container` is created, with the specifics configured as in
`docker-compose.yml` (port-forwarding, volume mirroring with the host OS etc.).

Check that your container is up and running:

```bash
$ sudo docker ps
```

You should see something like this:

```shell
CONTAINER ID        IMAGE               COMMAND               CREATED             STATUS              PORTS                    NAMES
44315329363a        gim_cv        "tail -f /dev/null"   29 seconds ago      Up 19 seconds       0.0.0.0:8888->8888/tcp   gim_cv_container
```
###



### disable firewall (in case of problems on CentOS)

If, on Linux, you run into the issue that the containers have no internet
(eg apt update fails), do:

```bash
sudo systemctl disable firewalld
```

```bash
systemctl stop docker
systemctl start docker
```

### Attach to the container

Now attach to the container to run code in the environment:

```bash
docker exec -it gim_cv_container /bin/bash
```

Press ctrl+D or type 'exit' to exit.

## Working in docker enviroment

Activate the environment with `conda activate gim_cv_gpu`.

## Install gim_cv as python package

Install `gim_cv` as a python package in editable mode. Inside the
conda environment, do:

```bash
$ pip install -e .
```

## Running training and inference

Deployment of the framework has been simplified and the nomenclature is as follows:

+ Place the training RGB images in the `TRAIN/rasters/` directory which is accessible from the main project repository (/home/root/)

+ Place the corresponding training labels in the `TRAIN/masks/` directory which is accessible from the main project repository (/home/root/)

+ Place the RGB images to be used for inference, in the `INFER/` directory which is accessible from the main project repository (/home/root/)

+ The model will be saved in the `MODELS/` directory with the format *Model_name+checkpoint_uuid*

+ Once the inference section is launched, the prediction maps will be generated and saved under the directory `PREDICTIONS/` with the format *Model+model_name+checkpoint_uuid+Infer+Image_name*

### Training script

To launch the training of a model, you can specify the training dataset label, spatial resolution of the target area imagery and additional features. These parameters can be specified in accordance with the instructions below.
```bash
$ python bin/train_segmentalist.py -d train_tif -tsr 0.4 -sag -ds -pp -dcbam -ecbam -ot -l dice_coeff_loss -ep 200
```

### Inference script

Once you have a model trained with satisfactory performance, you can use it to make some predictions on predifined location.
The rasters of the area of interest can be stored in the default inference directory as instructed or in a specific location that you need to set with the `infer_data_tif_path` parameter in `config.yml`. The location of the model's repository and the path to be used to store the segmentation map can also be specified respectively with the parameters `models_path` and `predictions_data_tif_path`.
```bash
$ python bin/run_inference_segmentalist.py -td train_tif -d infer_tif -w 512 -l dice_coeff_loss 


