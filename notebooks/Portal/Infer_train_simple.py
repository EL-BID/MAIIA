#!/usr/bin/env python
# coding: utf-8

# ## ENTRY POINT FOR THE INFORMAL SETTLEMENT DELINEATION FRAMEWORK

# This notebook demonstrates how you can launch training script, compare model performance and load the weights of a trained model, then use this for inference and store the segmentation maps.

# In[ ]:


import sys
import gim_cv.config as cfg
import numpy as np

sys.path.append('../../bin')
from bin import utility
from utility import *


# The current framework interface has been simplified and the nomenclature is as follows:
# 
# + Place the training RGB images in the *TRAIN/rasters/* directory which is accessible from the main project repository (/home/root/)
#   
#   
# + Place the corresponding training labels in the *TRAIN/masks/* directory which is accessible from the main project repository (/home/root/)
#   
#   
# + Place the RGB images to be used for inference, in the *INFER/* directory which is accessible from the main project repository (/home/root/)
#   
#   
# + The model will be saved in the *MODELS/* directory with the format *Model_name+checkpoint_uuid+current_date*
#   
#   
# + Once the inference section is launched, the prediction maps will be generated and saved under the directory *PREDICTIONS/* with the format *Model+model_name+checkpoint_uuid+Infer+Image_name+current_date*
# To launch the training of a model, you can specify the model path and spatial resolution of the target area imagery. Both parameters are set to their default values in accordance with the instructions above.
# In[ ]:


train_model(models_dir = str(cfg.models_path), target_spatial_resolution = 0.4)

# Once you have a model trained with satisfactory performance, you can use it to make some predictions on predifined location.
# The rasters of the area of interest can be stored in the default inference directory as instructed or in a specific location that you need to set with the *inference_dir* parameter. The location of the model's repository and the path to be used to store the segmentation map can also be specified respectively with the parameters *model_dir* and *predictions_dir*.
# In[ ]:


infer_from_model(predictions_dir = cfg.predictions_data_tif_path, inference_dir = str(cfg.infer_data_tif_path), models_dir = str(cfg.models_path), inference_window_size = 512)

