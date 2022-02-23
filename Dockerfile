FROM nvidia/cuda:10.1-base

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# install linux libraries

# add-apt-repo
#RUN apt-get install software-properties-common
# latest gdal
#RUN add-apt-repository ppa:ubuntugis/ppa
RUN apt-get update --fix-missing && \
    apt-get install -y software-properties-common wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion build-essential vim && \
    apt-get clean
RUN add-apt-repository ppa:ubuntugis/ppa
RUN apt-get install -y gdal-bin
# install anaconda (python 3) https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set the default directory in container
WORKDIR /home/root/

# copy contents of this repo in (to the container WORKDIR)
COPY envs/*.yml ./envs/
# not necessary
#COPY data ./data

# Create a conda environment from the environment specification file
RUN conda env create -f ./envs/env-static.yml
# gim_cv_gpu_env.yml, env-static.yml
# ... and for pytorch
#RUN conda env create -f ./envs/pytorch_test_gpu_env.yml

# get jupyter themes for nicer notebook colours
RUN pip install jupyterthemes

# create dask config
COPY .dask_config.yaml ~/.dask/config.yaml

# change themes and adjust cell width
RUN jt -t chesterish -cellw 95%

# add an alias "jn" to launch a jupyter-notebook process on port 8888
RUN echo 'alias jn="jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root --port=8888"' >> ~/.bashrc

# ports for jupyter notebook
EXPOSE 8888
# dask cluster
EXPOSE 8787
# tensorboard
EXPOSE 8686



CMD [ "/bin/bash" ]
