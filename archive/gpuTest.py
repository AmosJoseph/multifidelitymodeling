#!/usr/bin/env python
# coding: utf-8

# # Multi-Fidelity Surrogate Model Comparison - Kriging vs. Deep Neural Net (DNN) 
# This code seeks to compare the data compression, accuracy, portability, and evaluation time for two different types of surrogate modeling techniques: Kriging and Deep Neural Nets (DNN). First, we will build single fidelity models based on the RANS data (our high fidelity data set), then we will build single fidelity models based on local methods (low fidelity data set). Finally, we will build multi-fidelity models combining the data from both models. 
# 
# Our goal is to beat the performance of the single fidelity model, and also potentially explore how much of the high fidelity data is needed--maybe we can match the performace of the single fidelity model, but with significantly less data. 

# # Make Decisions About What You'd Like To Do
# In the below cell, make some choices about where you're executing this program. 

# In[404]:


print('GPU and import test started')


# In[405]:

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'# if you wish to supress TensorFlow debug info, set to 3
# import code
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6); plt.rcParams['axes.grid'] = True; plt.rcParams['figure.dpi'] = 300; plt.rcParams['axes.labelsize'] = 'xx-large';
from tensorflow.keras import layers
# %config InlineBackend.figure_format = 'svg'

import tensorflow as tf
# import warnings
# import sklearn
# import math
import datetime
# import copy
import time
import random
# import scipy
import GPUtil
# import skopt
import keras_tuner as kt

from sklearn import preprocessing
from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split#, GridSearchCV, RepeatedStratifiedKFold
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras import mixed_precision
from scipy.stats import qmc
from tabulate import tabulate

print('Imports successful')


# # GPU Details

# In[407]:


print("="*40, "Mixed Precision Policy", "="*40)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("="*40, "GPU Details", "="*40)
gpus = GPUtil.getGPUs()
list_gpus = []
for gpu in gpus:
    # get the GPU id
    gpu_id = gpu.id
    # name of GPU
    gpu_name = gpu.name
    # get % percentage of GPU usage of that GPU
    gpu_load = f"{gpu.load*100}%"
    # get free memory in MB format
    gpu_free_memory = f"{gpu.memoryFree}MB"
    # get used memory
    gpu_used_memory = f"{gpu.memoryUsed}MB"
    # get total memory
    gpu_total_memory = f"{gpu.memoryTotal}MB"
    # get GPU temperature in Celsius
    gpu_temperature = f"{gpu.temperature} °C"
    gpu_uuid = gpu.uuid
    list_gpus.append((
        gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
        gpu_total_memory, gpu_temperature, gpu_uuid
    ))

print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory", "temperature", "uuid")))

print(device_lib.list_local_devices())


# In[ ]:


programPause = input("Press the <ENTER> key to continue...")


