ARG base_image=tensorflow/tensorflow:2.8.0-jupyter

FROM $base_image AS app

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_VERSION=1.1.3

# Temporal workaround: Replace apt sources for NVIDIA and CUDA libraries and
# their signing keys using official CUDA keyring.
# Ref: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
  && rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/tensorRT.list \
  && apt-get install -y --no-install-recommends wget \
  && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
  && dpkg -i cuda-keyring_1.0-1_all.deb \
  && rm cuda-keyring_1.0-1_all.deb

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libproj-dev \
    libspatialindex-dev \
    proj-bin \
    python3-dev \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

ADD pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.create false && \
  poetry install --no-interaction --no-ansi

COPY . .

RUN pip install .

ENV MPLCONFIGDIR=/tmp/.config/matplotlib
