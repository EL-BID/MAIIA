.PHONY: build build-gpu run run-gpu clean

TAGNAME := maiia
VERSION := latest
GPU_VERSION := $(VERSION)-gpu
BASE_IMAGE     := tensorflow/tensorflow:2.8.0-jupyter
GPU_BASE_IMAGE := tensorflow/tensorflow:2.8.0-gpu-jupyter
RUN_OPTS := -u $(shell id -u):$(shell id -g) \
  -v $(realpath ./notebooks):/tf/notebooks \
  -v $(realpath ./data):/app/data/ \
  -p 8888:8888

TERM := /bin/bash

run: build
	docker run -ti --rm $(RUN_OPTS) $(TAGNAME):$(VERSION) $(TERM)

run-jupyter: build
	docker run -ti --rm $(RUN_OPTS) $(TAGNAME):$(VERSION)

run-gpu: build-gpu
	docker run -ti --rm --gpus all --runtime=nvidia $(RUN_OPTS) $(TAGNAME):$(GPU_VERSION) $(TERM)

run-jupyter-gpu: build-gpu
	docker run -ti --rm --gpus all --runtime=nvidia $(RUN_OPTS) $(TAGNAME):$(GPU_VERSION)

build:
	docker build --force-rm --build-arg base_image=$(BASE_IMAGE) -t $(TAGNAME):$(VERSION) .

build-gpu:
	docker build --force-rm --build-arg base_image=$(GPU_BASE_IMAGE) -t $(TAGNAME):$(GPU_VERSION) .

clean:
	docker rmi $(TAGNAME):$(VERSION)
	docker rmi $(TAGNAME):$(GPU_VERSION)
