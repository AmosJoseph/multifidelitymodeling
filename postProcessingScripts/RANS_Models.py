#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#Supress TensorFlow debug info. We don't have a GPU or TPU, 
                                        #and if TensorFlow doesn't find one, it gets upset
import code
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6); plt.rcParams['axes.grid'] = True; plt.rcParams['figure.dpi'] = 300
get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'svg'
import tensorflow as tf
import warnings
import tqdm
import sklearn
import math
import datetime
import copy
import time
import random


from tqdm.keras import TqdmCallback
from sklearn import preprocessing
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import qmc


# # Multi-Fidelity Surrogate Model Comparison - Kriging vs. Deep Neural Net (DNN) 
# This code seeks to compare the data compression, accuracy, portability, and evaluation time for two different types of surrogate modeling techniques: Kriging and Deep Neural Nets (DNN). First, we will build single fidelity models based on the RANS data (our high fidelity data set), then we will build single fidelity models based on local methods (low fidelity data set). Finally, we will build multi-fidelity models combining the data from both models. 
# 
# Our goal is to beat the performance of the single fidelity model, and also potentially explore how much of the high fidelity data is needed--maybe we can match the performace of the single fidelity model, but with significantly less data. 

# ### Functions Block - Run First

# In[129]:


def latinHypercubeSample(dimensionParameterSpace, numPointsToSample,l_bounds,u_bounds,seed):
    """

    :param dimensionParameterSpace: 
    :param numPointsToSample: 
    :param l_bounds: 
    :param u_bounds: 
    :return:
    """
    sampler = qmc.LatinHypercube(d=dimensionParameterSpace, seed=seed)
    sample = sampler.random(n=numPointsToSample)
    sample.shape
    # Wall Temp, rho, free stream temp, free stream velocity
    lowFidelityInputPoints = qmc.scale(sample, l_bounds, u_bounds)
    return lowFidelityInputPoints

def variableChecker(stringToTest):
    if stringToTest in globals():
        print('Global variable')
    else: 
        print('Not global variable')

    if stringToTest in locals():
        print('Local variable')
    else: 
        print('Not local variable')

#Oblique Perfect Gas Function 
def perfgas_oblique(M,V1,T1,P1,rho_1,a1,theta):
    
    gamma = 1.4 # perfect gas
    R_specific = 287.058
    cp1 = 1.005 #KJ/Kg*K, air at 251K
    #For our initial guess at beta, let's use the M>>1 approximation. Although
    #our flow M>>1, this code will calculate a more accurate turn angle than the approximation. 
    b_init = ((gamma + 1) / 2)*theta

    #We'll use Newton's method, which requires the function and the function's 
    #derivative, included below as "f" and "fp"
    #  tan(theta) = 2cot(beta)*(M^2sin^2(beta) - 1)/(M^2(gamma + cos(2beta) + 1)

    b = np.zeros((11,1))
    i = 1
    b[i] = b_init
    
    for i in range(1,10):
        b[i+1] = b[i] - ((2*(1/np.tan(b[i]))*(M**2*(np.sin(b[i]))**2-1))/(M**2*(gamma + np.cos(2*b[i]))+2)-np.tan(theta))
        (((4*M**2*np.sin(2*b[i])*(1/np.tan(b[i]))*(M**2*np.sin(b[i])**2-1))/((M**2*(np.cos(2*b[i])+gamma)+2)**2))
           + (4*M**2*np.cos(b[i])**2 - 2*(1/np.cos(b[i]))**2*(M**2*(np.sin(b[i]))**2-1))/(M**2*(np.cos(2*b[i])+gamma)+2))

            
    beta = b[10]
    # beta_deg = np.rad2deg(b[10])

    M1 = M
    M2 = np.sqrt(((1+((gamma-1)/2)*(M1**2)*(np.sin(beta)**2)) / (gamma*(M1**2)*(np.sin(beta)**2)-((gamma-1)/2)))
        * (1/(np.sin(beta-theta)**2)))
    m_ratio = M2/M1

    temp_ratio = 1 + ((2*(gamma-1))/((gamma+1)**2))*(((M1**2)*(np.sin(beta)**2) - 1)/((M1**2)*(np.sin(beta)**2)))*(gamma*(M1**2)*(np.sin(beta)**2)+1)
    T2 = temp_ratio*T1

    H2 = cp1*T2*1000

    #Using the relation T2/T1 = (a2/a1)^2, we can also solve for the ratio of
    #a2/a1 
    a_ratio = np.sqrt(temp_ratio)
    a2 = a_ratio*a1

    rho_ratio = ((gamma+1)*(M1**2)*(np.sin(beta)**2))/((gamma-1)*(M1**2)*(np.sin(beta)**2)+2)
    rho2 = rho_ratio*rho_1

    v_ratio = m_ratio* a_ratio
    V2 = v_ratio*V1

    p_ratio = rho_ratio*temp_ratio
    P2 = p_ratio*P1

    T01 = T1*(1+ ((gamma-1)/2)*(M1**2))
    T02 = T2*(1+ ((gamma-1)/2)*(M2**2))

    total_p_ratio = p_ratio * ((1+((gamma-1)/2)*(M2**2))**(gamma/(gamma-1)))/((1+((gamma-1)/2)*(M1**2))**(gamma/(gamma-1)))
    P01 = P1* ((1 +((gamma-1)/2)*(M1**2))**((gamma)/(gamma-1)))
    P02 = P01*total_p_ratio
    
    return H2, V2, T2, P2, rho2, beta, M2,a2, T01, T02, P01, P02

def mu_suth(T):
    mu_ref = 1.8e-5
    T_ref = 300
    mu = mu_ref*((T/T_ref)**0.7)
    return mu

# Neural Network Functions
def build_model_single_fidelity(inputData, outputData, numOutputData, layer1size, layer2Size, rate, reg):
    """

    :param inputData: input parameters/features (only one (!!!!) of the arrays, not all of them)
    :param outputData: the outputs the NN is fitting (only one (!!!!) of the arrays, not all of them)
    :param numOutputData: number of outputs the NN will have
    :param layer1size: reference size for building the NN model
    :param layer2size: reference size for building the NN model
    :param rate: learning rate
    :param reg: L2 regularization value to drop weights
    :return:
    """

    inputlayershape = int(len(inputData[0,:]))
    outputlayershape = int(len(outputData[0,:]))
    
    T_w = tf.keras.Input(shape=(inputlayershape,), name = 'walltemp')
    rho = tf.keras.Input(shape=(inputlayershape,), name = 'freestream_density')
    T_inf = tf.keras.Input(shape=(inputlayershape,), name = 'freestream_temp')
    u_inf = tf.keras.Input(shape=(inputlayershape,), name = 'freestream_velocity')
    
    inputs = [T_w, rho , T_inf,u_inf]
    
    features = layers.Concatenate()(inputs)
    
    x = tf.keras.layers.Dense(layer1size,activation=tf.nn.tanh,
                kernel_regularizer=keras.regularizers.l2(reg),
                kernel_initializer = tf.keras.initializers.GlorotUniform()
                )(features)
    
    x = tf.keras.layers.Dense(layer2size,activation=tf.nn.tanh,
                kernel_regularizer=keras.regularizers.l2(reg),
                kernel_initializer = tf.keras.initializers.GlorotUniform()
                )(x)
    
#     x = tf.keras.layers.Dense(layer3_size,activation=tf.nn.tanh,
#                 kernel_regularizer=keras.regularizers.l2(reg),
#                 kernel_initializer = tf.keras.initializers.GlorotUniform()
#                 )(x)
    
    qw_layer = tf.keras.layers.Dense(outputlayershape,activation=tf.nn.leaky_relu,
                      kernel_regularizer= keras.regularizers.l2(reg),
                      kernel_initializer = tf.keras.initializers.GlorotUniform(),
                                     name = 'wall_heat_transfer')(x)
    
    
    P_layer = tf.keras.layers.Dense(outputlayershape, activation=tf.nn.leaky_relu,
                  kernel_regularizer= keras.regularizers.l2(reg),
                  kernel_initializer = tf.keras.initializers.GlorotUniform()
                                    , name='wall_pressure')(x)
    
    outputs = [qw_layer, P_layer]
    
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=rate),
             loss = "mse",
             metrics = [tf.keras.metrics.MeanSquaredError(),
                       tf.keras.metrics.RootMeanSquaredError(),])
                       # "mae"])
    tf.keras.utils.plot_model(model,show_shapes=True)
    return model

#Building the model
def build_model_multi_fidelity(input_data, output_data, ref_size, rate, reg):
    """

    :param input_data: input parameters/features
    :param output_data: outputs the NN is fitting
    :param ref_size: reference size for building the NN model
    :param rate: learning rate
    :param reg: L2 regularization value to drop weights
    :return:
    """
    inputlayershape = int(len(input_data[0,:]))
    outputlayershape = int(len(output_data[0,:]))

    layer1_size = ref_size
#     layer2_size = ref_size
#     layer3_size = ref_size
    inputs = tf.keras.Input(shape=(inputlayershape,))
    
    x = tf.keras.layers.Dense(layer1_size,activation=tf.nn.tanh,
                kernel_regularizer=keras.regularizers.l2(reg),
                kernel_initializer = tf.keras.initializers.GlorotUniform()
                )(inputs)
    x = tf.keras.layers.Dense(layer1_size,activation=tf.nn.tanh,
                kernel_regularizer=keras.regularizers.l2(reg),
                kernel_initializer = tf.keras.initializers.GlorotUniform()
                )(x)
    
#     x = tf.keras.layers.Dense(layer3_size,activation=tf.nn.tanh,
#                 kernel_regularizer=keras.regularizers.l2(reg),
#                 kernel_initializer = tf.keras.initializers.GlorotUniform()
#                 )(x)

    # Repeat the "x = ...(x) " pattern to generate more layers if desired   
   
    qw_layer = tf.keras.layers.Dense(outputlayershape,activation=tf.nn.leaky_relu,
                      kernel_regularizer= keras.regularizers.l2(reg),
                      kernel_initializer = tf.keras.initializers.GlorotUniform(),
                                     name = 'wall_heat_transfer')(x)
    
    
    P_layer = tf.keras.layers.Dense(outputlayershape, activation=tf.nn.leaky_relu,
                  kernel_regularizer= keras.regularizers.l2(reg),
                  kernel_initializer = tf.keras.initializers.GlorotUniform()
                                    , name='wall_pressure')(x)
    
    outputs = [qw_layer, P_layer]

    # outputs = tf.keras.layers.Dense(outputlayershape, activation=tf.nn.leaky_relu,
    #                   kernel_regularizer= keras.regularizers.l2(reg),
    #                   kernel_initializer = tf.keras.initializers.GlorotUniform())(x)
    
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=rate),
             loss = "mse",
             metrics = [tf.keras.metrics.MeanSquaredError(),
                       tf.keras.metrics.RootMeanSquaredError(),])
                       # "mae"])
    tf.keras.utils.plot_model(model,show_shapes=True)
    return model

#training the model 

# tqdm_callback = tfa.callbacks.TQDMProgressBar()

def train_model_all_fidelity(model, input_data, output_data, numEpochs, myBatchSize, validSplit):
    callbacks_list = [
        # keras.callbacks.EarlyStopping(
        # monitor = "root_mean_squared_error",mode="min",
        # patience=1000, verbose=2),
        TqdmCallback(verbose=1)
#         keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_path.keras,
#         monitor="mean_squared_error")
#         keras.callbacks.LearningRateScheduler(scheduler
#         )
    ]
    
    history = model.fit(x=input_data,
                       y=output_data,
                       batch_size=myBatchSize,
                       epochs=numEpochs,
                       callbacks = callbacks_list,
                       verbose=False,
                       shuffle=False,
                       validation_split=validSplit,
                       use_multiprocessing=True)
    epochs = history.epoch
    return epochs, history.history


# # Import data from CFD runs. 
# If you have not run the post-processing scripts housed in the top folder, run those first. The below code assumes that all us3d output has been post-processed

# In[40]:


# Import the *input* data, i.e. what we tell the RANS code the initial state in front of our wedge is. 

path = "/home/tyty/Desktop/CFD_Files/AFRL_Scholars/1st_Round_Ls"
os.chdir(path)
# Provide the file name that holds your input data: 
fileName = '1st_Round_LS.cvs'

# First thing is to read in the data file
data = pd.read_csv(fileName)

# Convert the pandas dataframe to numpy and transpose
inputData = pd.DataFrame(data).to_numpy()
inputData = inputData.T

# Iteratively assign the input data names. The enumerate function has a second output, which I've stored 
# here with "i" that indexes your data for you, so you don't need to use an initialized counter (i=0)
# and then count with an i += 1 at the end. This section of code uses the locals()[] to turn a string
#into a variable so we can iteratively create variables.
inputVarNames = [
    'WallTemp',
    'Density',
    'Temperature',
    'Velocity',
]

l_bounds = []
u_bounds = []

for i, name in enumerate(inputVarNames):
    name = 'input' + name
    locals()[name] = inputData[i,:].reshape(-1,1)
    minInputVal = np.min(locals()[name])
    maxInputVal = np.max(locals()[name])
    l_bounds.append(minInputVal)
    u_bounds.append(maxInputVal)
    print( name + ' range from: ' + str(minInputVal) + ' to ' + str(maxInputVal) + '\n') 


# ### Input Data
# 
# A quick explanation of our models' input data 
# 
# inputWallTemp = wall temperature of double cone
# 
# inputDensity = free stream density
# 
# inputTemperature = free stream temperature
# 
# inputVelocity = free stream velocity
# 
# These values are all chosen by the user, and the values are fed to US3D to generate CFD results for the entire flowfield. To keep the data and the model light, we're only going to analyze the "wall" values., which in this case will be pressure and heat transfer (maybe eventually the shear stresses, too). 

# ### Generate Low-Fidelity Space Sample Points 

# In[41]:


dimensionParameterSpace = 4 # wall temp, free stream temp, free stream density, free stream velocity
numPointsToSample = 4000 
seed = 5 # seed with a number for reproducibility. It will be the same LHS sampling every time the function
        # is run using seed = x. 

lowFidelityInputPoints = latinHypercubeSample(dimensionParameterSpace, 
                                              numPointsToSample,l_bounds,u_bounds,seed)


# In[42]:


# Code that plots Low Fideltiy Input Space vs. High Fidelity Input Space  

# wallTempLimits = (400,450)
# densityLimits = (0.2,0.3)

# matplotlib.rcParams['figure.figsize'] = (5,5)
# plt.rcParams['legend.facecolor'] = 'black'
# plt.scatter(lowFidelityInputPoints[:,0],lowFidelityInputPoints[:,1], s=4, label = 'Low Fidelity Space Sample Points')
# plt.scatter(inputWallTemp[:],inputDensity[:], label = 'High Fidelity Space Sample Points', s=150)
# plt.xlabel("Wall Temp",fontsize='xx-large')
# plt.ylabel("Density",fontsize='xx-large')
# plt.grid()
# lgnd = plt.legend(loc='upper right', fontsize='large')
# for i in range(0,2):
#     lgnd.legendHandles[i]._sizes = [100]
    
# for text in lgnd.get_texts():
#     text.set_color("white")
# plt.xlim(wallTempLimits)
# plt.ylim(densityLimits)


# plt.rcParams["figure.figsize"] = (20,10)
# fig, axs = plt.subplots(2, 3)
# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# fig.patch.set_facecolor('white')


# axs[0,0].scatter(lowFidelityInputPoints[:,0],lowFidelityInputPoints[:,1], s=1, label = 'Low Fidelity Space Sample Points')
# axs[0,0].scatter(inputWallTemp[:],inputDensity[:], label = 'High Fidelity Space Sample Points')
# axs[0,0].set_xlabel("Wall Temp")
# axs[0,0].set_ylabel("Density")
# axs[0,0].grid()
# axs[0,0].legend()


# In[43]:


# Import the *output* data, i.e. what the RANS code does with our input file. 
# Take these from the files 
path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
os.chdir(path)

outputVarNames = [
    'x',
    'y',
    'z',
    'qw',
    'taux',
    'tauy',
    'tauz',
    'tauw',
    'yplus',
    'T',
    'p',
    'x_cc',
    'y_cc',
    'z_cc'
]
for name in outputVarNames:
    locals()[name] = None
    pickleName = './' + name + '.pkl'
    F = open(pickleName, 'rb')
    locals()[name] = pickle.load(F)
    locals()[name] = locals()[name].T
    # Uncomment the below line if you'd like confirmation that your data made it 
    # print(locals()[name].shape)

pickleName = './' + 'conn' + '.pkl'
F = open(pickleName, 'rb')    
connDict = pickle.load(F)

# Pruning our list of output variables to what we'll need for training, testing, and validation
outputVarNames = [
    'qw',
    'p',
]

for name in outputVarNames:
    max_element = str(round(np.max(locals()[name]),2))
    min_element = str(round(np.min(locals()[name]),2))
    print(name + ' has been imported! Min:' + min_element + '. Max:' + max_element)


# # Windowing the Data

# In[44]:


# Do you desire to "window" the data and remove the pressure/heat transfer values from the nose region? If so, set Window to "True"
# If you desire to window the data, also select the location in which you'd like to do that. 
Window = True 

if Window:
    caseWePick = 389
    xWindowStart = 0.3
    xIndex = np.argsort(x_cc[0])
    x_cc_sorted = x_cc[:,xIndex]
    idxWindowStart = (np.abs(x_cc_sorted[0,:] - xWindowStart)).argmin()
    
    for name in outputVarNames:
        sortedName = name + '_sorted'
        windowedName = name + '_windowed'
        locals()[sortedName] = locals()[name][:,xIndex]
        locals()[windowedName] = locals()[sortedName][:,idxWindowStart:]
    
    x_cc_windowed = x_cc_sorted[:,idxWindowStart:]
    
    plt.rcParams["figure.figsize"] = (20,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
    fig.patch.set_facecolor('white')


    axs[0].scatter(x_cc_sorted[caseWePick,idxWindowStart:], p_sorted[caseWePick,idxWindowStart:],s=30, label = 'Windowed Data')
    axs[0].scatter(x_cc_sorted[caseWePick,:], p_sorted[caseWePick,:],s=3, label = 'Full Data')
    axs[0].axvline(x=xWindowStart, linewidth=1, color='k', label = 'Chosen Start of Window')
    axs[1].scatter(x_cc_sorted[caseWePick,idxWindowStart:], qw_sorted[caseWePick,idxWindowStart:],s=30, label = 'Windowed Data')
    axs[1].scatter(x_cc_sorted[caseWePick,:], qw_sorted[caseWePick,:],s=3, label = 'Full Data')
    axs[1].axvline(x=xWindowStart, linewidth=1, color='k', label = 'Chosen Start of Window')

    axs[0].set_title("Pressure Window Results")
    axs[0].set_ylabel("Pressure (Pa)")
    axs[0].grid()
    axs[1].set_title("Heat Transfer Window Results")
    axs[1].set_ylabel('Heat Transfer (w/m^2')
    axs[1].grid()

    axs[0].set_xlabel("x Distance Along Cone")
    axs[1].set_xlabel("x Distance Along Cone")
    
    axs[0].legend()
    axs[1].legend()
    
# Do you desire to overwrite the original output data with the windowed and sorted output data? If so, set windowedOverwrite
# to "true." This will only run if Window is also true
windowedOverwrite = True

if Window and windowedOverwrite and (qw_windowed.any() != None):
    for name in outputVarNames:
        sortedName = name + '_sorted'
        windowedName = name + '_windowed'
        locals()[name] = None
        locals()[name] = locals()[windowedName]
        locals()[sortedName] = None
        locals()[windowedName] = None


# ### Input Data
# 
# A quick explanation of our models' output data 
# 
# qw = heat transfer at the wall
# 
# p = pressure at the wall
# 
# We won't be training the model using the x,y,z,tau,yplus, T, or the cell center locations. These will be used after the model is trained for analysis purposes. Note that the "T", or wall temperature value, was fixed by the CFD code as the thermal boundary condition. If you take a look inside the output T array, you'll find that it's all the same number as the input wall temperature. 

# # Generate Low Fidelity Data - Shock Expansion, Eckert's Reference Temp
# 
# ## Progress Notes
# 
# 10MAR22 - Just completed stitching the two pressure predictions together--newtonian overpredicts the peak pressure
# Next, stitch together the heat transfer values using the pressure predictions generated here^^^^ - there may be a nice way to turn the "T2" vectors into the 7deg and 40deg sections, kind of like how the low fidelity pressure vector is stitched together. would make the code a bit cleaner, and prevent a duplication of calculations.
# 
# 11MAR22 - corrected the inviscid shock relations issue--the second domain sees a 33deg turn angle, not a 40 deg turn angle. discovered conical shock relations. generated entire cone of pressure and heat transfer. compared results--seems to be some structure to it... some correlation. 
# 
# 15MAR22 - goal: train models on LF data. complete
# 
# 16MAR22 - goal: clean up the LF code, compare LF data to predictions
# 
# To do:
# - plot the cp's side by side (subplots)
# - remove outlier case from the data entirely
# <!-- I think this may not be good. seems to underpredict even worse: using conical shock relations instead of oblique shock relations - https://gist.github.com/gusgordon/3fa0a80e767a34ffb8b112c8630c5484#file-supersonic_shock_solver-py -->
# - turn this whole script into a function to call in the RANS model, generate the data in one line. 
# - windowing code cell has bug where if the data is already windowed it gets unhappy. put a try/except or an if statement in there somewhere. 
# - remove all of the x_cc_windowed 
# - make it so the predicted values for LF kriging and NN don't overwrite each other. right now they have the same name

# In[20]:


## Input Conditions, low fidelity data generation

gamma = 1.4 # perfect gas
R_specific = 287.058
cp1 = 1.005 #KJ/Kg*K, air at 251K

T_inf = inputTemperature
T_w = inputWallTemp
rho_inf = inputDensity
u_inf = inputVelocity
a_inf = np.sqrt(gamma*R_specific*T_inf)
M_inf = u_inf/a_inf
P_inf = rho_inf*R_specific*T_inf
mu_inf = mu_suth(T_inf)

theta  = np.full((400,1),np.deg2rad(7))

inputDataObliqueShock = M_inf,u_inf,T_inf,P_inf,rho_inf,a_inf,theta

[*temp] = map(perfgas_oblique, M_inf,u_inf,T_inf,P_inf,rho_inf,a_inf,theta)
obliqueShockResults = np.array(temp)

outputLocalMethodVarName = [
    'H2', 
    'V2', 
    'T2', 
    'P2', 
    'rho2',
    'beta',
    'M2',
    'a2', 
    'T01', 
    'T02',
    'P01',
    'P02'
]

for i, name in enumerate(outputLocalMethodVarName):
    locals()[name] = obliqueShockResults[:,i]


# In[49]:


## ---- Pressure Coefficient ----

# Shock Expansion for 7deg Section

shockAngle = beta

cp_ShockExpansionTheory = (4/(gamma+1))*(np.sin(shockAngle)**2 - (1/(M_inf**2)))
cp_newtonian_coneAngle = 2*(np.sin(theta)**2)

xPressureWindowStart = 0.5
xPressureWindowEnd = 2.0
xPressureWindowMid = 0.5

xSpotBegin = (np.abs(x_cc_windowed[0,:] - xPressureWindowStart)).argmin()
xSpotEnd = (np.abs(x_cc_windowed[0,:] - xPressureWindowEnd)).argmin()
xSpotNoMean = (np.abs(x_cc_windowed[0,:] - xPressureWindowMid)).argmin()

PressureForCPActual = p[:,xSpotNoMean].reshape(-1,1)
cp_actual = (PressureForCPActual - P_inf)/ (0.5*rho_inf*(u_inf**2))
cp_actual[389] = None # takes care of that one bad point

#Shock Expansion For 40deg Section

T_inf2 = T2
T_w = inputWallTemp
rho_inf2 = rho2
u_inf2 = M2*a2
a_inf2 = a2
M_inf2 = M2
P_inf2 = P2
mu_inf2 = mu_suth(T2)
theta2  = np.full((400,1),np.deg2rad(33))

[*temp] = map(perfgas_oblique, M_inf2,u_inf2,T_inf2,P_inf2,rho_inf2,a_inf2,theta2)
obliqueShockResults = np.array(temp)

outputLocalMethodVarName = [
    'H3', 
    'V3', 
    'T3', 
    'P3', 
    'rho3',
    'beta2',
    'M3',
    'a3', 
    'T02', 
    'T03',
    'P02',
    'P03'
]

for i, name in enumerate(outputLocalMethodVarName):
    locals()[name] = obliqueShockResults[:,i]


# In[51]:


xPressureWindowLeft = 2.353056 # elbow location
xPressureWindowRight = 2.5039961 # end of cone

xPressureWindowMid = 2.4
xSpotLeft = (np.abs(x_cc_windowed[0,:] - xPressureWindowLeft)).argmin()
xSpotRight = (np.abs(x_cc_windowed[0,:] - xPressureWindowRight)).argmin()

meanPressure40DegConeSection = np.median(p[:,xSpotLeft:xSpotRight], axis = 1).reshape(-1,1)
cp_actual2 = (meanPressure40DegConeSection - P_inf2)/ (0.5*rho_inf2*(u_inf2**2))
cp_actual2[389] = None # takes care of that one bad point

cp_newtonian_coneAngle2 = 2*(np.sin(theta2)**2)
shockAngle2 = beta2
cp_ShockExpansionTheory2 = (4/(gamma+1))*(np.sin(shockAngle2)**2 - (1/(M_inf2**2)))

p_SE_7deg = cp_ShockExpansionTheory*(0.5*rho_inf*(u_inf**2)) + P_inf
p_Newtonian_40deg = cp_newtonian_coneAngle2*(0.5*rho_inf2*(u_inf2**2)) + P_inf2
p_SE_40deg = cp_ShockExpansionTheory2*(0.5*rho_inf2*(u_inf2**2)) + P_inf2

xSpotElbow = (np.abs(x_cc_windowed[0,:] - xPressureWindowLeft)).argmin()

p_lf_7deg = np.tile(p_SE_7deg, xSpotElbow+1)
p_lf_40deg_newt = np.tile(p_Newtonian_40deg, xSpotRight - xSpotLeft)
p_lf_40deg = np.tile(p_SE_40deg, xSpotRight - xSpotLeft)
p_lowFidelity_SE = np.concatenate((p_lf_7deg, p_lf_40deg), axis=1)
p_lowFidelity_Newt = np.concatenate((p_lf_7deg, p_lf_40deg_newt), axis=1)

normalizedHFPressure = p/P03
normalizedLFPressure = p_lowFidelity_SE/P03
normalizedLFPressureNewt = p_lowFidelity_Newt/P03


# ## Plotting Pressure Coefficient Results

# In[47]:


# caseWePick = random.randint(0,400)

# plt.plot(x_cc_windowed[0,:],normalizedLFPressure[caseWePick,:], label = 'Low Fidelity (SE)', color = 'k')
# plt.plot(x_cc_windowed[0,:],normalizedLFPressureNewt[caseWePick,:], label = 'Low Fidelity (Newtonian)', color = 'b', linestyle='-.')
# plt.scatter(x_cc_sorted[caseWePick,idxWindowStart:], normalizedHFPressure[caseWePick,:],s=5, label = 'Truth Data (RANS)', color='r')
# plt.legend()
# plt.title('Case Number ' + str(caseWePick))
# plt.xlabel('x, axial location (meters)')
# plt.ylabel('$P/P_{03}$')
# plt.grid()

# plt.rcParams["figure.figsize"] = (5,5)
# plt.grid()
# plt.scatter(M_inf,cp_actual, label = 'RANS Truth CP', s=3)
# plt.scatter(M_inf,cp_ShockExpansionTheory, label = 'S-E Theory CP', s=3)
# plt.scatter(M_inf,cp_newtonian_coneAngle, label = 'Newtonian Theory CP', s=3)
# lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=20)
# plt.xlabel("Mach Number",fontsize='xx-large')
# plt.ylabel("Pressure Coefficient",fontsize='xx-large')
# plt.ylim([0,0.12])
# lgnd = plt.legend(loc='upper right', fontsize='x-large', markerscale = 5)
# # for i in range(0,7):
# #     lgnd.legendHandles[i]._sizes = [50]

# TURN THIS INTO A SUBLPLOT, side by side

# plt.rcParams["figure.figsize"] = (7,5)
# plt.rcParams['figure.dpi'] = 300
# plt.grid()
# plt.scatter(M_inf,cp_actual2, label = 'RANS Truth CP', s=3)
# plt.scatter(M_inf,cp_ShockExpansionTheory2, label = 'S-E Theory CP', s=3)
# plt.scatter(M_inf,cp_newtonian_coneAngle2, label = 'Newtonian Theory CP', s=3)
# plt.scatter(M_inf,cp_actual, label = 'RANS Truth CP', s=3)
# plt.scatter(M_inf,cp_ShockExpansionTheory, label = 'S-E Theory CP', s=3)
# plt.scatter(M_inf,cp_newtonian_coneAngle, label = 'Newtonian Theory CP', s=3)
# lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=20)
# plt.rcParams['legend.facecolor'] = 'white'
# plt.xlabel("Mach Number",fontsize='xx-large')
# plt.ylabel("Pressure Coefficient",fontsize='xx-large')
# lgnd = plt.legend(loc='upper right', fontsize='x-large', markerscale = 5)


# # Eckert's Reference Temperature

# In[52]:


## ---- Eckert's Reference Temperature, Cone Example ----

caseWePick = random.randint(0,399)
Pr = 0.72
recovFactor = np.sqrt(Pr)
xSpotEndArtificial = x_cc_windowed[0,:].shape[0] - xSpotElbow

x_FrontCone = x_cc_windowed[0,:xSpotElbow]
x_RearCone = x_cc_windowed[0,:xSpotEndArtificial] - 0.25

T_star = 0.5*(T2 + T_w) + .22*recovFactor*(T02 - T2)
rho_star = P2/ (R_specific*T_star)
mu_star = mu_suth(T2)
u2 = M2 * a2
Re_coeff = rho_star*u2/mu_star
Re_x = Re_coeff * x_FrontCone
cone_factor = np.sqrt(3)
cH_coeff = (cone_factor*0.332)/((Pr**(2/3))*(Re_coeff**(.5)))
cH_star = cH_coeff / x_FrontCone**(1/2)
T_r = T2 + recovFactor*(T02 - T2)
q_dot_FrontCone = rho_star*u2*cp1*1000*(T_r - T_w)*cH_star

T_star2 = 0.5*(T3 + T_w) + .22*recovFactor*(T03 - T3)
rho_star2 = P3/ (R_specific*T_star2)
mu_star2 = mu_suth(T3)
u3 = M3* a3
Re_coeff2 = rho_star2*u2/mu_star2
Re_x2 = Re_coeff2 * x_RearCone
cH_coeff2 = (cone_factor*0.332)/((Pr**(2/3))*(Re_coeff2**(.5)))
cH_star2 = cH_coeff2 / x_RearCone**(1/2)
T_r2 = T3 + recovFactor*(T03 - T3)
q_dot_RearCone = rho_star2*u3*cp1*1000*(T_r2 - T_w)*cH_star2

q_lowFidelity = np.concatenate((q_dot_FrontCone, q_dot_RearCone), axis=1)

plt.semilogy(x_cc_windowed[0,:],q_lowFidelity[caseWePick,:],label = 'Low Fidelity Heat Transfer')
plt.semilogy(x_cc_windowed[0,:],qw[caseWePick,:], label = 'Truth (RANS) Heat Transfer')
plt.grid()
plt.xlabel('Location Along Wedge Wall, $x$ (meters)')
plt.ylabel('Heat Transfer Rate ($q_w$)')
plt.legend()


# # Low Fidelity Data Scaling

# In[53]:


LFoutputVarNames = [
    'q_lowFidelity',
    'p_lowFidelity_SE',
]

inputTrainingData = []
inputTrainingNames = []

print('Input Data (stored in list inputTrainingData):\n')
for i, name in enumerate(inputVarNames):
    ScalerName = name + '_InputScaler'
    ScaledName = name + '_Scaled'
    InputDataName = 'input' + name
    locals()[ScalerName] = None
    locals()[ScalerName] = preprocessing.StandardScaler()
    locals()[ScaledName] = locals()[ScalerName].fit_transform(globals()[InputDataName])
    inputTrainingData.append(locals()[ScaledName])
    inputTrainingNames.append(ScaledName)
    max_element = str(round(np.max(locals()[ScaledName]),2))
    min_element = str(round(np.min(locals()[ScaledName]),2))
    print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

outputTrainingData = []
outputTrainingNames = []

print('\nOutput Data (stored in list outputTrainingData):\n')
for i, name in enumerate(LFoutputVarNames):
    ScalerName = name + '_OutputScaler'
    ScaledName = name + '_Scaled'
    OutputDataName = name
    locals()[ScalerName] = None
    locals()[ScalerName] = preprocessing.StandardScaler()
    locals()[ScaledName] = locals()[ScalerName].fit_transform(globals()[OutputDataName])
    outputTrainingData.append(locals()[ScaledName])
    outputTrainingNames.append(ScaledName)
    max_element = str(round(np.max(locals()[ScaledName]),2))
    min_element = str(round(np.min(locals()[ScaledName]),2))
    print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

print(str(np.shape(inputTrainingData)))
print(str(np.shape(outputTrainingData)))
print(inputTrainingNames)
print(outputTrainingNames)


# In[54]:


LFinputDataConcat = np.concatenate((WallTemp_Scaled,Density_Scaled,Temperature_Scaled,Velocity_Scaled),axis=1)
LFoutputDataConcat = np.concatenate((q_lowFidelity_Scaled,p_lowFidelity_SE_Scaled),axis=1)


# # LF Kriging

# In[55]:


# Kernel options include: Compound, Constant, Dot Product, Exponentiation, ExpSineSquared, 
# Product, Rational Quadratic, RBF, Sum, White

# Benchmark Function 1
kernel1 = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(1.0, (1e-8, 1e2)) # Works great for Benchmark Function 1

#Benchmark Function 2
kernel2 = 66.3*RBF(0.1, (1e-8, 1e2)) #seems to work well for discontinuities, so we'll try for Benchmark Function 2

#Benchmark Function 3
kernel3 = kernel2

# Below are the different kernels (and combinations of kernels) that Dr. Reasor used in his code

#kernel = ConstantKernel(1.0) + Matern(length_scale=0.1, nu=3/2) #+ WhiteKernel(noise_level=1)
#kernel = 1.0*Matern(length_scale=0.1, nu=1.5)
#kernel = RationalQuadratic()
#kernel = Matern(length_scale=0.1, nu=2.5)  #
#kernel = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(0.1, (1e-8, 1e2))
#kernel = Cdatetime.dateonstantKernel(1.0, (1e-3, 1e3))*RBF(1.0, (1e-2, 1e2))


# In[56]:


LF_krig = None
LF_krig = gaussian_process.GaussianProcessRegressor(kernel=kernel2,n_restarts_optimizer=0)
LF_krig.fit(LFinputDataConcat, LFoutputDataConcat)


# In[58]:


save = True

if save:
    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    os.chdir(path)

    model = LF_krig
    filename = 'LF_krig_'
    dt = str(datetime.date.today())
    ext = '.sav'
    filename += dt + ext
    pickle.dump(model, open(filename, 'wb'))


# In[60]:


LF_krig_prediction = LF_krig.predict(LFinputDataConcat)


# In[61]:


#Now we'll need to un-concatenate and scale the output predictions
outputDataSplit = int(LFoutputDataConcat.shape[1]/len(LFoutputVarNames))
indexStart = 0
indexEnd = outputDataSplit

outputPredictedData = []

for i, name in enumerate(LFoutputVarNames):
    OutputDataName = name + '_predicted'
    locals()[OutputDataName] = LF_krig_prediction[:,indexStart:indexEnd]
    indexStart += outputDataSplit
    indexEnd += outputDataSplit
    
    ScalerName = name + '_OutputScaler'
    locals()[OutputDataName] = locals()[ScalerName].inverse_transform(globals()[OutputDataName])
    outputPredictedData.append(locals()[ScaledName])
    max_element = str(round(np.max(locals()[OutputDataName]),2))
    min_element = str(round(np.min(locals()[OutputDataName]),2))
    print(OutputDataName + ' has been inverse transformed! It is called ' + OutputDataName + 
          '. Min:' + min_element + '. Max:' + max_element)


# In[64]:


## Plotting Kriging predictions vs. actual low fidelity data

# desiredNumCasesForPlot = 15
# casePlotRange= np.arange(0,400,int((400/desiredNumCasesForPlot)))

# plt.rcParams["figure.figsize"] = (10,5)
# fig, axs = plt.subplots(1, 2)
# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
# fig.patch.set_facecolor('white')

  
# for i, case in enumerate(casePlotRange):
#     labelstr = 'Case: ' + str(case)
#     axs[0].scatter(q_lowFidelity_predicted[case,:],q_lowFidelity[case,:], s=1, label = labelstr )
#     axs[1].scatter(p_lowFidelity_SE_predicted[case,:],p_lowFidelity_SE[case,:], s=1, label = labelstr)


    
# # axs[0].scatter(qw_predicted[389,:],qw[389,:], s=1, label = 'Case 389' )
# # axs[1].scatter(p_predicted[389,:],p[389,:], s=1)

# axs[0].set_title("Kriging Heat Transfer Predicitions vs Actual")
# axs[1].set_title("Kriging Pressure Predictions vs. Actual")
# axs[0].grid()
# axs[1].grid()
# axs[0].set_ylabel("True Value")
# axs[0].set_xlabel("Predicted Heat Transfer")
# axs[1].set_xlabel("Predicted Pressure")

# # for i in range(0,2):
# # axs[0].legend().legendHandles[0]._sizes = [100]


# # fig.savefig('Krig_predict.png', bbox_inches='tight')


# In[67]:


caseWePick = random.randint(0,399)

plt.rcParams["figure.figsize"] = (10,5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
fig.patch.set_facecolor('white')

axs[0].semilogy(x_cc_windowed[0,:],q_lowFidelity_predicted[caseWePick,:],
                label = 'Low Fidelity Heat Transfer Predicitions', linestyle = '-.', color = 'red', zorder=3)
axs[0].semilogy(x_cc_windowed[0,:],q_lowFidelity[caseWePick,:],
                label = 'Truth (ERT) Heat Transfer',color='black', linestyle='solid',
                linewidth=4)
axs[0].grid()

axs[1].semilogy(x_cc_windowed[0,:],p_lowFidelity_SE_predicted[caseWePick,:],
                label = 'Low Fidelity Pressure Predicition', linestyle = '-.', color = 'red',zorder=3)
axs[1].semilogy(x_cc_windowed[0,:],p_lowFidelity_SE[caseWePick,:], 
                label = 'Truth (Shock Expansion) Pressure',color='black', linestyle='solid',
                linewidth=4)

axs[1].grid()
# xlim([min(x)-.1 max(x)+.1])
axs[1].set_xlabel('Radial Distance, $x$ (meters)')
axs[0].set_ylabel('Heat Transfer Rate ($q_w$)')

axs[1].set_xlabel('Radial Distance, $x$ (meters)')
axs[1].set_ylabel('Pressure (Pascals)')

axs[0].legend()
axs[1].legend()


# # Low Fidelity Neural Network 

# In[70]:


#Would you like to train a new model? 
trainNN = True

#Would you like to save your model? 
saveNN, saveHistoryandEpochs = True, True

#Would you like to load a previously trained model ?
loadNN, loadHistoryandEpochs = False , False

#If you have trained a model in this same workspace as well as loaded a model, would you like to overwrite with your loaded model? 
#If you have not trained a model, but loaded a model, then leave this "True"
if not trainNN and loadNN: 
    overwrite = True
    
#Initialize hyperparamters for Single Fidelity Model
if trainNN: 
    layer1size = 32 
    layer2size = 64
    numOutputData = 2
    rate = 1.0e-3 #From Deep Learning w/ Python (Chollet)
    # rate = 5.0e-2 # used by Dr. Reasor
    reg = 1.0e-6
    numEpochs = 1000
    myBatchSize = None
    validSplit = None


# In[71]:


if trainNN: 

    numCases = str(len(WallTemp_Scaled[:,0]))
    inputArraySize = str(WallTemp_Scaled.shape)
    outputArraySize = str(q_lowFidelity_Scaled.shape)
    numInput = str(len(inputVarNames))
    numOutput = str(len(LFoutputVarNames))
    singleOutput = str(len(q_lowFidelity_Scaled[0,:]))

    print('We are using %s cases. Meaning each input array is %s, each output array is %s \n' % 
          (numCases, inputArraySize, outputArraySize))
    print('Note: The network, as depicted below, will take data from one case at a time. Therefore,'
         'the input data will be %s 1x1 arrays, and the output will be %s 1x%s arrays \n \n \n' % 
          (numInput, numOutput, singleOutput ))
    LF_NN = None #sometimes remnants of previously trained models can hang around, it's best 
                 #to clear the variable first 
    LF_NN = build_model_single_fidelity(WallTemp_Scaled, q_lowFidelity_Scaled, numOutputData, 
                                        layer1size, layer2size, rate, reg)
    tf.keras.utils.plot_model(LF_NN,show_shapes=True)


# In[72]:


tf.keras.utils.plot_model(LF_NN,show_shapes=True)


# In[73]:


if trainNN:

    LF_NN_epochs = None
    LF_NN_history = None

    LF_NN_epochs, LF_NN_history = train_model_all_fidelity(
        LF_NN, inputTrainingData, outputTrainingData, numEpochs, myBatchSize, validSplit)


# In[74]:


if saveNN:

    # The try/except syntax catches the error if we've already created the directory. 
    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    kerasFolderName = 'LF_NN_'
    dt = str(datetime.date.today())
    kerasFolderName += dt

    try: 
        os.chdir(path)
        mkdirPath = os.path.join(path, kerasFolderName)
        os.mkdir(mkdirPath)
    except: 
        pass 

    # We won't be saving the NN in the same way that we save the GP model. Keras likes to save all kind of other stuff, with no intuitive way to change the file names.
    # For version control purposes (really, to make sure we're not overwriting old models), we'll create a new folder for each saved model with the date included. 

    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    path += '/' + kerasFolderName
    os.chdir(path)

    model = LF_NN
    model.save(path)
    
if saveHistoryandEpochs:
    epochsDict = LF_NN_epochs
    historyDict = LF_NN_history

    epochsFilename = 'LF_NN_epochs'
    historyFilename = 'LF_NN_history'
    dt = str(datetime.date.today())
    ext = '.pkl'
    epochsFilename += '_' + dt + ext
    historyFilename += '_' + dt + ext
    pickle.dump(epochsDict, open(epochsFilename, 'wb'))
    pickle.dump(historyDict, open(historyFilename, 'wb'))


# ## Plot Training MSE for LF NN

# In[75]:


mse_history_qw = LF_NN_history["wall_heat_transfer_mean_squared_error"]
mse_history_P = LF_NN_history["wall_pressure_mean_squared_error"]

# Input the desired epoch range

epochRangeBegin = 0 
epochRangeEnd = 5000

plt.plot(range(1,len(mse_history_qw[epochRangeBegin:epochRangeEnd]) + 1),
         mse_history_qw[epochRangeBegin:epochRangeEnd],
         label="Heat Transfer Predicition MSE",linestyle="-", color="k")
plt.plot(range(1,len(mse_history_P[epochRangeBegin:epochRangeEnd]) + 1),
         mse_history_P[epochRangeBegin:epochRangeEnd],
         label="Pressure Predicition MSE",linestyle="-.", color="b")
plt.title("Single Fidelity Neural Network Predicition Performance ")
plt.legend(loc=0)
plt.grid()

minError = np.min(LF_NN_history["wall_heat_transfer_mean_squared_error"])
MSE = np.sqrt(minError)
print('Mean Squared Error %0.7f percent' % (MSE*100, ))


# ## LF NN Predictions

# In[76]:


[q_lowFidelity_predicted, p_lowFidelity_SE_predicted] = LF_NN.predict(inputTrainingData)


outputPredictedData = []

print('\nPredicted Output Data (stored in list outputPredictedData):\n')
for i, name in enumerate(LFoutputVarNames):
    ScalerName = name + '_OutputScaler'
    OutputDataName = name + '_predicted'
    locals()[OutputDataName] = locals()[ScalerName].inverse_transform(globals()[OutputDataName])
    outputPredictedData.append(locals()[ScaledName])
    max_element = str(round(np.max(locals()[OutputDataName]),2))
    min_element = str(round(np.min(locals()[OutputDataName]),2))
    print(OutputDataName + ' has been inverse transformed! It is called ' +
          OutputDataName + '. Min:' + min_element + '. Max:' + max_element)


# In[78]:


desiredNumCasesForPlot = 15
casePlotRange= np.arange(0,400,int((400/desiredNumCasesForPlot)))

plt.rcParams["figure.figsize"] = (10,5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
fig.patch.set_facecolor('white')

  
for i, case in enumerate(casePlotRange):
    labelstr = 'Case: ' + str(case)
    axs[0].scatter(q_lowFidelity_predicted[case,:],q_lowFidelity[case,:], s=1, label = labelstr )
    axs[1].scatter(p_lowFidelity_SE_predicted[case,:],p_lowFidelity_SE[case,:], s=1, label = labelstr)


    
# axs[0].scatter(qw_predicted[389,:],qw[389,:], s=1, label = 'Case 389' )
# axs[1].scatter(p_predicted[389,:],p[389,:], s=1)

axs[0].set_title("NN Heat Transfer Predicitions vs Actual")
axs[1].set_title("NN Pressure Predictions vs. Actual")
axs[0].grid()
axs[1].grid()
axs[0].set_ylabel("True Value")
axs[0].set_xlabel("Predicted Heat Transfer")
axs[1].set_xlabel("Predicted Pressure")


# In[77]:


caseWePick = random.randint(0,399)

plt.rcParams["figure.figsize"] = (10,5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
fig.patch.set_facecolor('white')

axs[0].semilogy(x_cc_windowed[0,:],q_lowFidelity_predicted[caseWePick,:],
                label = 'Low Fidelity Heat Transfer Predicitions', linestyle = '-.', color = 'red', zorder=3)
axs[0].semilogy(x_cc_windowed[0,:],q_lowFidelity[caseWePick,:],
                label = 'Truth (ERT) Heat Transfer',color='black', linestyle='solid',
                linewidth=4)
axs[0].grid()

axs[1].semilogy(x_cc_windowed[0,:],p_lowFidelity_SE_predicted[caseWePick,:],
                label = 'Low Fidelity Pressure Predicition', linestyle = '-.', color = 'red',zorder=3)
axs[1].semilogy(x_cc_windowed[0,:],p_lowFidelity_SE[caseWePick,:], 
                label = 'Truth (Shock Expansion) Pressure',color='black', linestyle='solid',
                linewidth=4)


axs[1].grid()
# xlim([min(x)-.1 max(x)+.1])
axs[1].set_xlabel('Radial Distance, $x$ (meters)')
axs[0].set_ylabel('Heat Transfer Rate ($q_w$)')

axs[1].set_xlabel('Radial Distance, $x$ (meters)')
axs[1].set_ylabel('Pressure (Pascals)')

axs[0].legend()
axs[1].legend()

#################################################################

## Similar code but with nicer plots (maybe?)

# case = 286 #Select the run that you would like to plot
# sliceVal = 25 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter
# xUpperLimit = 2.55
# xLowerLimit = None

# # inputWallTemp, inputDensity , inputTemperature , inputVelocity 
# # q_lowFidelity, p_lowFidelity_SE
    
# textstr = '\n'.join((
#     'Wall Temp: ' + str(round(inputWallTemp[case].item(), 2)) + ' K',
#     'Freestream Temp: ' + str(round(inputTemperature[case].item(), 2)) + ' K',
#     'Freestream Density: ' + str(round(inputDensity[case].item(), 2)) + ' kg/m3',
#     'Freestream Velocity: ' + str(round(inputVelocity[case].item(), 2)) + ' m/s'
#     ))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# plt.rcParams["figure.figsize"] = (15,3.5)
# fig, axs = plt.subplots(1, 2)
# fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.5)
# fig.patch.set_facecolor('white')

# # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
# axs[0].plot(x_cc_sorted[0,idxWindowStart:], q_lowFidelity[case,:]/10000, color='firebrick', linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')
# axs[0].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], q_lowFidelity_predicted[case,::sliceVal]/10000, c='white',
#                zorder=3,edgecolors='black', marker='D', s=70, label='Neural Network Predicition')
# # axs[0].set_title("Predicted Heat Transfer, Single Fidelity Only",fontsize='x-large')
# axs[0].set_ylabel("Heat Transfer (W/cm^2)", fontsize='x-large')

# # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
# axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_lowFidelity_SE[case,:]*1e-3, color='black', linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')
# axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], p_lowFidelity_SE_predicted[case,::sliceVal]*1e-3, c='white',
#                zorder=3,edgecolors='black', marker='D', s=70, label='Neural Network Predicition')
# axs[1].set_title("Predicted Pressure", fontsize='x-large')
# axs[1].set_ylabel("Pressure (kPa)", fontsize='x-large')

# for i in np.arange(0,len(axs)):
#     # axs[i].grid()
#     # axs[i].legend().get_frame().set_edgecolor('b')
#     # axs[i].legend().get_frame().set_linewidth(10)
#     axs[i].legend(fontsize='x-large', edgecolor='k')
#     axs[i].set_xlabel('x-distance Along Cone (meters)',fontsize='x-large')
#     axs[i].set_xlim([xLowerLimit,xUpperLimit])
#     # axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
#     #     verticalalignment='top', bbox=props)


# # High Fidelity Data Scaling (using sklearn's Standard.Scaler)
# 
# In order to model effectively, we'll want to scale our training data appropriately. In this case, we're using sklearn's ```preprocessing.StandardScaler()```. The module standardizes features by removing the mean and scaling to unit variance. The standard score of a sample x is calculated as: ```z = (x - u) / s```, where ```u``` is the mean of the training samples or zero if ```with_mean=False```, and ```s``` is the standard deviation of the training samples or one if ```with_std=False```.
# 
# This is included in the comments of the below code block, but keep in my that each data set needs its own distinct scaler object. If you don't create a scaler object for each data set, you won't be able to inverse transform the data properly. 
# 
# The code in the next cell looks ugly, but it is a more robust way to do what the code directly below this sentence is doing. First, we make sure our new variable is not pointing to a Scaler object with the "None." Then, we assign the variable a Scaler object. Lastly, we create a new np.array with our scaled dating using the ".fit_transform" function. 
# 
# ScalerName = None
# 
# ScalerName = preprocessing.StandardScaler()
# 
# ScaledData = ScalerName.fit_transform(original_data)

# In[6]:


inputTrainingData = []
inputTrainingNames = []

print('Input Data (stored in list inputTrainingData):\n')
for i, name in enumerate(inputVarNames):
    ScalerName = name + '_InputScaler'
    ScaledName = name + '_Scaled'
    InputDataName = 'input' + name
    locals()[ScalerName] = None
    locals()[ScalerName] = preprocessing.StandardScaler()
    locals()[ScaledName] = locals()[ScalerName].fit_transform(globals()[InputDataName])
    inputTrainingData.append(locals()[ScaledName])
    inputTrainingNames.append(ScaledName)
    max_element = str(round(np.max(locals()[ScaledName]),2))
    min_element = str(round(np.min(locals()[ScaledName]),2))
    print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

outputTrainingData = []
outputTrainingNames = []

print('\nOutput Data (stored in list outputTrainingData):\n')
for i, name in enumerate(outputVarNames):
    ScalerName = name + '_OutputScaler'
    ScaledName = name + '_Scaled'
    OutputDataName = name
    locals()[ScalerName] = None
    locals()[ScalerName] = preprocessing.StandardScaler()
    locals()[ScaledName] = locals()[ScalerName].fit_transform(globals()[OutputDataName])
    outputTrainingData.append(locals()[ScaledName])
    outputTrainingNames.append(ScaledName)
    max_element = str(round(np.max(locals()[ScaledName]),2))
    min_element = str(round(np.min(locals()[ScaledName]),2))
    print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

print(str(np.shape(inputTrainingData)))
print(str(np.shape(outputTrainingData)))
print(inputTrainingNames)
print(outputTrainingNames)


# In[7]:


inputDataConcat = np.concatenate((WallTemp_Scaled,Density_Scaled,Temperature_Scaled,Velocity_Scaled),axis=1)
outputDataConcat = np.concatenate((qw_Scaled,p_Scaled),axis=1)


# In[10]:


## Only use these guys if you want your model to converge faster--for demo purposes or something. 
## I used this data intitially to make sure that my program was working correctly. 

# slicedInputDataConcat = inputDataConcat[:200] # taking only the first 200 cases, to make the model speed up a bit
# slicedOutputDataConcat = outputDataConcat[:200]


# # Kriging (Gaussian Process, or GP)

# ## Gaussian Process Kernels
# What kernels do/how to select kernels:
# 
# [The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)
# 

# In[13]:


# Kernel options include: Compound, Constant, Dot Product, Exponentiation, ExpSineSquared, 
# Product, Rational Quadratic, RBF, Sum, White

# Benchmark Function 1
kernel1 = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(1.0, (1e-8, 1e2)) # Works great for Benchmark Function 1

#Benchmark Function 2
kernel2 = 66.3*RBF(0.1, (1e-8, 1e2)) #seems to work well for discontinuities, so we'll try for Benchmark Function 2

#Benchmark Function 3
kernel3 = kernel2

# Below are the different kernels (and combinations of kernels) that Dr. Reasor used in his code

#kernel = ConstantKernel(1.0) + Matern(length_scale=0.1, nu=3/2) #+ WhiteKernel(noise_level=1)
#kernel = 1.0*Matern(length_scale=0.1, nu=1.5)
#kernel = RationalQuadratic()
#kernel = Matern(length_scale=0.1, nu=2.5)  #
#kernel = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(0.1, (1e-8, 1e2))
#kernel = Cdatetime.dateonstantKernel(1.0, (1e-3, 1e3))*RBF(1.0, (1e-2, 1e2))


# ## Gaussian Process (using sklearn's gaussian_process)
# 
# *Building our multi-fidelity kriging the ol' fashioned way* 
# 
# This module provides the user control over the hyperparameters (specifically the kernels, n restarts optimizer), but is a bit more abstract that GPy, which is why it was chosen over other GP modules (like GPy). 
# 
# kernel: class-kernel instance, default=None
# The kernel specifying the covariance function of the GP. If None is passed, the kernel ConstantKernel(1.0, constant_value_bounds="fixed" * RBF(1.0, length_scale_bounds="fixed") is used as default. Note that the kernel hyperparameters are optimized during fitting unless the bounds are marked as “fixed”.
# 
# n_restarts_optimizer: class-int, default=0
# The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood. The first run of the optimizer is performed from the kernel’s initial parameters, the remaining ones (if any) from thetas sampled log-uniform randomly from the space of allowed theta-values. If greater than 0, all bounds must be finite. Note that n_restarts_optimizer == 0 implies that one run is performed.

# ## Building the GP Model and Fitting the Data

# In[14]:


load = True
save = False


# In[15]:


if not load:
    RANS_krig = None
    RANS_krig = gaussian_process.GaussianProcessRegressor(kernel=kernel2,n_restarts_optimizer=0)


# Multi-input, multi-output Kriging
# 
# "At the heart of your issue lies something rarely mentioned (or even hinted at) in practice and in relevant tutorials: Gaussian Process regression with multiple outputs is highly non-trivial and still a field of active research. Arguably, scikit-learn cannot really handle the case, despite the fact that it will superficially appear to do so, without issuing at least some relevant warning."
# 
# https://stackoverflow.com/questions/50185399/multiple-output-gaussian-process-regression-in-scikit-learn

# In[16]:


if not load:
    RANS_krig.fit(inputDataConcat, outputDataConcat)


# ## Saving or loading your models
# 
# The larger your data gets, and the longer the train time gets, the more you may be interested in saving/loading a pre-trained model. 

# In[17]:


## Saving your model
if save:
    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    os.chdir(path)

    model = RANS_krig
    filename = 'RANS_krig_'
    dt = str(datetime.date.today())
    ext = '.sav'
    filename += dt + ext
    pickle.dump(model, open(filename, 'wb'))


# In[18]:


# some time later... load the model from disk
if load:
    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    os.chdir(path)

    filename = 'RANS_krig_2022-02-18.sav'
    desiredLoadedModelName = 'RANS_krig_loaded'

    locals()[desiredLoadedModelName] = pickle.load(open(filename, 'rb'))


# ## Overwrite current loaded model? 
# 
# If you trained a model in the running kernel, but wish to overwrite that model with the one you loaded, use the cell below.

# In[19]:


## If you want to overwrite your model with a loaded model, overwrite = True
## If you don't want to overwrite your model, overwrite = False

overwrite = True

if overwrite:
    RANS_krig = None
    RANS_krig = RANS_krig_loaded


# ## Inverse Transform

# In[20]:


inputDataConcat = np.concatenate((WallTemp_Scaled,Density_Scaled,Temperature_Scaled,Velocity_Scaled),axis=1)
outputDataConcat = np.concatenate((qw_Scaled,p_Scaled),axis=1)


# In[21]:


# Now that we have a prediction, we'll need inverse transform and see how well we did. 

RANS_krig_prediction = RANS_krig.predict(inputDataConcat)


# In[22]:


#Now we'll need to un-concatenate and scale the output predictions
outputDataSplit = int(outputDataConcat.shape[1]/len(outputVarNames))
indexStart = 0
indexEnd = outputDataSplit

outputPredictedData = []

for i, name in enumerate(outputVarNames):
    OutputDataName = name + '_predicted'
    locals()[OutputDataName] = RANS_krig_prediction[:,indexStart:indexEnd]
    indexStart += outputDataSplit
    indexEnd += outputDataSplit
    
    ScalerName = name + '_OutputScaler'
    locals()[OutputDataName] = locals()[ScalerName].inverse_transform(globals()[OutputDataName])
    outputPredictedData.append(locals()[ScaledName])
    max_element = str(round(np.max(locals()[OutputDataName]),2))
    min_element = str(round(np.min(locals()[OutputDataName]),2))
    print(OutputDataName + ' has been inverse transformed! It is called ' + OutputDataName + '. Min:' + min_element + '. Max:' + max_element)


# ## Analyze Our Predictions

# In[24]:


desiredNumCasesForPlot = 10
casePlotRange= np.arange(0,400,int((400/desiredNumCasesForPlot)))

plt.rcParams["figure.figsize"] = (10,5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
fig.patch.set_facecolor('white')

  
for i, case in enumerate(casePlotRange):
    labelstr = 'Case: ' + str(case)
    axs[0].scatter(qw_predicted[case,:],qw[case,:], s=1, label = labelstr )
    axs[1].scatter(p_predicted[case,:],p[case,:], s=1, label = labelstr)


    
axs[0].scatter(qw_predicted[389,:],qw[389,:], s=1, label = 'Case 389' )
axs[1].scatter(p_predicted[389,:],p[389,:], s=1)

axs[0].set_title("Heat Transfer Predicitions vs Actual")
axs[1].set_title("Pressure Predictions vs. Actual")
axs[0].grid()
axs[1].grid()
axs[0].set_ylabel("True Value")
axs[0].set_xlabel("Predicted Heat Transfer")
axs[1].set_xlabel("Predicted Pressure")

for i in range(0,2):
axs[0].legend().legendHandles[0]._sizes = [100]


fig.savefig('Krig_predict.png', bbox_inches='tight')


# In[8]:


case = 200 #Select the run that you would like to plot
sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

# inputWallTemp, inputDensity , inputTemperature , inputVelocity 
    
textstr = '\n'.join((
    'Wall Temp: ' + str(round(inputWallTemp[case].item(), 2)) + ' K',
    'Freestream Temp: ' + str(round(inputTemperature[case].item(), 2)) + ' K',
    'Freestream Density: ' + str(round(inputDensity[case].item(), 2)) + ' kg/m3',
    'Freestream Velocity: ' + str(round(inputVelocity[case].item(), 2)) + ' m/s'
    ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.rcParams["figure.figsize"] = (15,5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.5)
fig.patch.set_facecolor('white')

# plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw[case,:]/10000, color='firebrick', linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')
axs[0].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], qw_predicted[case,::sliceVal]/10000, c='white',
               zorder=3,edgecolors='black', marker='D', s=70, label='Kriging Predicition')
axs[0].set_title("Predicted Heat Transfer",fontsize='x-large')
axs[0].set_ylabel("Heat Transfer (W/cm^2)", fontsize='x-large')

# plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
axs[1].plot(x_cc_sorted[0,idxWindowStart:], p[case,:]*1e-3, color='black', linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')
axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], p_predicted[case,::sliceVal]*1e-3, c='white',
               zorder=3,edgecolors='black', marker='D', s=70, label='Kriging Predicition')
axs[1].set_title("Predicted Pressure", fontsize='x-large')
axs[1].set_ylabel("Pressure (kPa)", fontsize='x-large')

for i in np.arange(0,len(axs)):
    # axs[i].grid()
    axs[i].legend(fontsize='x-large')
    axs[i].set_xlabel('x (meters)',fontsize='x-large')
    axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


# # Artificial Neural Network
# Below are the functions that create and train the Neural Network models
# 
# The functions can become more robust by using *kwargs instead of layer1size, etc.... but for another day 

# ## Initialize Hyperparameters

# In[9]:


#Would you like to train a new model? 
trainNN = False

#Would you like to save your model? 
saveNN, saveHistoryandEpochs = False, False

#Would you like to load a previously trained model ?
loadNN, loadHistoryandEpochs = True , True

#If you have trained a model in this same workspace as well as loaded a model, would you like to overwrite with your loaded model? 
#If you have not trained a model, but loaded a model, then leave this "True"
if not trainNN and loadNN: 
    overwrite = True


# In[10]:


#Initialize hyperparamters for Single Fidelity Model
if trainNN: 
    layer1size = 32 
    layer2size = 64
    numOutputData = 2
    rate = 1.0e-3 #From Deep Learning w/ Python (Chollet)
    # rate = 5.0e-2 # used by Dr. Reasor
    reg = 1.0e-6
    numEpochs = 1000
    myBatchSize = None
    validSplit = None


# ## Build Model and Plot Graph

# In[11]:


if trainNN: 

    numCases = str(len(WallTemp_Scaled[:,0]))
    inputArraySize = str(WallTemp_Scaled.shape)
    outputArraySize = str(qw_Scaled.shape)
    numInput = str(len(inputVarNames))
    numOutput = str(len(outputVarNames))
    singleOutput = str(len(qw_Scaled[0,:]))

    print('We are using %s cases. Meaning each input array is %s, each output array is %s \n' % 
          (numCases, inputArraySize, outputArraySize))
    print('Note: The network, as depicted below, will take data from one case at a time. Therefore,'
         'the input data will be %s 1x1 arrays, and the output will be %s 1x%s arrays \n \n \n' % (numInput, numOutput, singleOutput ))
    RANS_NN = None #sometimes remnants of previously trained models can hang around, it's best to clear the variable first 
    RANS_NN = build_model_single_fidelity(WallTemp_Scaled, qw_Scaled, numOutputData, layer1size, layer2size, rate, reg)
    tf.keras.utils.plot_model(RANS_NN,show_shapes=True)


# ## Train the Model

# In[12]:


if trainNN:

    RANS_NN_epochs = None
    RANS_NN_history = None

    RANS_NN_epochs, RANS_NN_history = train_model_all_fidelity(
        RANS_NN, inputTrainingData, outputTrainingData, numEpochs, myBatchSize, validSplit)


# ## Save and Load Data
# Only run the below cell if you desire to save the model. We'll also save the history dict in a pickle file. 

# ### Save NN

# In[13]:


## If you want to overwrite your model with a loaded model, overwrite = True
## If you don't want to overwrite your model, overwrite = False
if saveNN:

    # The try/except syntax catches the error if we've already created the directory. 
    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    kerasFolderName = 'RANS_NN_'
    dt = str(datetime.date.today())
    kerasFolderName += dt

    try: 
        os.chdir(path)
        mkdirPath = os.path.join(path, kerasFolderName)
        os.mkdir(mkdirPath)
    except: 
        pass 

    # We won't be saving the NN in the same way that we save the GP model. Keras likes to save all kind of other stuff, with no intuitive way to change the file names.
    # For version control purposes (really, to make sure we're not overwriting old models), we'll create a new folder for each saved model with the date included. 

    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    path += '/' + kerasFolderName
    os.chdir(path)

    model = RANS_NN
    model.save(path)


# ### Save History and Epochs

# In[14]:


# Let's save the history and epoch data in a dictionary as well.
if saveHistoryandEpochs:
    epochsDict = RANS_NN_epochs
    historyDict = RANS_NN_history

    epochsFilename = 'RANS_NN_epochs'
    historyFilename = 'RANS_NN_history'
    dt = str(datetime.date.today())
    ext = '.pkl'
    epochsFilename += '_' + dt + ext
    historyFilename += '_' + dt + ext
    pickle.dump(epochsDict, open(epochsFilename, 'wb'))
    pickle.dump(historyDict, open(historyFilename, 'wb'))


# ### Load NN

# In[15]:


if loadNN: 
    # Loading the NN is a bit easier-- but you'll need to specify the path. An example path is included already, which will need to be edited if you wish to load a different model. 
    path = '/home/tyty/Desktop/CFD_Files/MFmodelTestData/RANS_NN_2022-02-26'
    os.chdir(path)
    loadedModelName = 'RANS_NN_loaded'

    locals()[loadedModelName] = keras.models.load_model(path)

### Load History and Epochs

if loadHistoryandEpochs:
    # some time later... load the model from disk

    # path variable should already be defined from when you loaded the NN model. If not, uncomment the 2 lines below. 
    # path = '/home/tyty/Desktop/CFD_Files/MFmodelTestData/RANS_NN_2022-02-11'
    # os.chdir(path)

    epochsFilename = 'RANS_NN_epochs_2022-02-26.pkl'
    historyFilename = 'RANS_NN_history_2022-02-26.pkl'

    desiredLoadedEpochsName = 'RANS_NN_epochs_loaded'
    desiredLoadedHistoryName = 'RANS_NN_history_loaded'

    locals()[desiredLoadedEpochsName] = pickle.load(open(epochsFilename, 'rb'))
    locals()[desiredLoadedHistoryName] = pickle.load(open(historyFilename, 'rb'))


# ## Overwrite current loaded model? 
# 
# If you trained a model in the running kernel, but wish to overwrite that model with the one you loaded, use the cell below.

# In[17]:


## If you want to overwrite your model with a loaded model, overwrite = True
## If you don't want to overwrite your model, leave overwrite = False

if overwrite:
    RANS_NN = None
    RANS_NN_epochs = None
    RANS_NN_history = None
    RANS_NN = RANS_NN_loaded
    RANS_NN_epochs = RANS_NN_epochs_loaded
    RANS_NN_history = RANS_NN_history_loaded


# In[18]:


RANS_NN.summary()


# ## Analyze Error (MSE) In High Fidelity Neural Network (RANS Data)

# In[19]:


mse_history_qw = RANS_NN_history["wall_heat_transfer_mean_squared_error"]
mse_history_P = RANS_NN_history["wall_pressure_mean_squared_error"]

# Input the desired epoch range

epochRangeBegin = 0 
epochRangeEnd = 5000

plt.plot(range(1,len(mse_history_qw[epochRangeBegin:epochRangeEnd]) + 1),
         mse_history_qw[epochRangeBegin:epochRangeEnd],
         label="Heat Transfer Predicition MSE",linestyle="-", color="k")
plt.plot(range(1,len(mse_history_P[epochRangeBegin:epochRangeEnd]) + 1),
         mse_history_P[epochRangeBegin:epochRangeEnd],
         label="Pressure Predicition MSE",linestyle="-.", color="b")
plt.title("Single Fidelity Neural Network Predicition Performance ")
plt.legend(loc=0)
plt.grid()

minError = np.min(RANS_NN_history["wall_heat_transfer_mean_squared_error"])
MSE = np.sqrt(minError)
print('Mean Squared Error %0.7f percent' % (MSE*100, ))


# ## Generate Predictions

# In[20]:


numCases = len(x_cc[:,0])

start = time.time()
[qw_predicted, p_predicted] = RANS_NN.predict(inputTrainingData)
end = time.time()
print('Prediction time per case: %0.7f seconds, %0.3f Hz ' % (((end-start)/numCases), 1/((end-start)/numCases)) )


# ## Inverse Transform Predictions

# In[21]:


# Now that we have a prediction, we'll need inverse transform and see how well we did. 

[qw_predicted, p_predicted] = RANS_NN.predict(inputTrainingData)

outputPredictedData = []

print('\nPredicted Output Data (stored in list outputPredictedData):\n')
for i, name in enumerate(outputVarNames):
    ScalerName = name + '_OutputScaler'
    OutputDataName = name + '_predicted'
    locals()[OutputDataName] = locals()[ScalerName].inverse_transform(globals()[OutputDataName])
    outputPredictedData.append(locals()[ScaledName])
    max_element = str(round(np.max(locals()[OutputDataName]),2))
    min_element = str(round(np.min(locals()[OutputDataName]),2))
    print(OutputDataName + ' has been inverse transformed! It is called ' + OutputDataName + '. Min:' + min_element + '. Max:' + max_element)


# ## Analyze Our Predictions
# For 20 cases, we have plotted the predicted value vs. the actual value. A perfect prediciton would be a straight line with a slope of 1. 

# In[22]:


desiredNumCasesForPlot = 15
casePlotRange= np.arange(0,400,int((400/desiredNumCasesForPlot)))

plt.rcParams["figure.figsize"] = (10,5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
fig.patch.set_facecolor('white')

for case in casePlotRange:
    caseStr = 'Case: ' + str(case)
    axs[0].scatter(qw_predicted[case,:],qw[case,:], s=1, label = caseStr)
    axs[1].scatter(p_predicted[case,:],p[case,:], s=1, label = caseStr)

axs[0].set_title("Heat Transfer Predicitions vs Actual")
axs[1].set_title("Pressure Predictions vs. Actual")
axs[0].grid()
axs[1].grid()
axs[0].set_ylabel("True Value")
axs[0].set_xlabel("Predicted Heat Transfer")
axs[1].set_xlabel("Predicted Pressure")

axs[0].legend(markerscale=8)

# for i in range(0,desiredNumCasesForPlot+1):
#     axs[0].legend().legendHandles[i]._sizes = [100]

fig.savefig('NN_predict.png', bbox_inches='tight')


# In[28]:


case = 286 #Select the run that you would like to plot
sliceVal = 25 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter
xUpperLimit = 2.55
xLowerLimit = 2.2

# inputWallTemp, inputDensity , inputTemperature , inputVelocity 
    
textstr = '\n'.join((
    'Wall Temp: ' + str(round(inputWallTemp[case].item(), 2)) + ' K',
    'Freestream Temp: ' + str(round(inputTemperature[case].item(), 2)) + ' K',
    'Freestream Density: ' + str(round(inputDensity[case].item(), 2)) + ' kg/m3',
    'Freestream Velocity: ' + str(round(inputVelocity[case].item(), 2)) + ' m/s'
    ))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

plt.rcParams["figure.figsize"] = (15,3.5)
fig, axs = plt.subplots(1, 2)
fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.5)
fig.patch.set_facecolor('white')

# plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw[case,:]/10000, color='firebrick', linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')
axs[0].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], qw_predicted[case,::sliceVal]/10000, c='white',
               zorder=3,edgecolors='black', marker='D', s=70, label='Neural Network Predicition')
# axs[0].set_title("Predicted Heat Transfer, Single Fidelity Only",fontsize='x-large')
axs[0].set_ylabel("Heat Transfer (W/cm^2)", fontsize='x-large')

# plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
# axs[1].plot(x_cc_sorted[0,idxWindowStart:], p[case,:]*1e-3, color='black', linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')
# axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], p_predicted[case,::sliceVal]*1e-3, c='white',
#                zorder=3,edgecolors='black', marker='D', s=70, label='Neural Network Predicition')
# axs[1].set_title("Predicted Pressure", fontsize='x-large')
# axs[1].set_ylabel("Pressure (kPa)", fontsize='x-large')

for i in np.arange(0,len(axs)):
    # axs[i].grid()
    # axs[i].legend().get_frame().set_edgecolor('b')
    # axs[i].legend().get_frame().set_linewidth(10)
    axs[i].legend(fontsize='x-large', edgecolor='k')
    axs[i].set_xlabel('x-distance Along Cone (meters)',fontsize='x-large')
    axs[i].set_xlim([xLowerLimit,xUpperLimit])
    # axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
    #     verticalalignment='top', bbox=props)


# ## Validation Split
# 
# Now let's try the same thing, but hold out a validation data set and compare our model's performance on the training data to the validation data. 

# In[24]:


# generate indices for the shuffled data set. It will generate an array of numbers 1:arg in a random order. The "arg" here is
# the length of the training input data, which is also the total number of cases run. 
indices_permutation = np.random.permutation(len(WallTemp_Scaled))

shuffledInputTrainingData = []

print('Shuffled Input Data (stored in list shuffledInputTrainingData):')
for i, name in enumerate(inputVarNames):
    ScaledName = name + '_Scaled'
    ShuffledName = ScaledName + '_Shuffled'
    # shuffled_inputs = inputs[indices_permutation]
    locals()[ShuffledName] = None
    locals()[ShuffledName] = locals()[ScaledName][indices_permutation]
    shuffledInputTrainingData.append(locals()[ShuffledName])
print('Shape of data: ' + str(np.shape(shuffledInputTrainingData)) + '\n')

shuffledOutputTrainingData = []

print('Shuffled Output Data (stored in list shuffledOutputTrainingData):')
for i, name in enumerate(outputVarNames):
    ScaledName = name + '_Scaled'
    ShuffledName = ScaledName + '_Shuffled'
    # shuffled_inputs = inputs[indices_permutation]
    locals()[ShuffledName] = None
    locals()[ShuffledName] = locals()[ScaledName][indices_permutation]
    shuffledOutputTrainingData.append(locals()[ShuffledName])
print('Shape of data: ' + str(np.shape(shuffledOutputTrainingData)))


# In[159]:


# Now that our data is shuffled, let's set aside some of our data for validation. 

desiredValidationSplit = 0.2 #set this variable to the desired percentage of data set aside for validation. Author chose 20% here
numValidationSamples = math.floor(desiredValidationSplit*len(WallTemp_Scaled))

## This is not right... but you're getting close 
valInputData = shuffledInputTrainingData[1][:numValidationSamples,:]
valOutputData = shuffledOutputTrainingData[1][:numValidationSamples,:]
trainInputData = shuffledInputTrainingData[1][numValidationSamples:,:]
trainOutputData = shuffledOutputTrainingData[1][numValidationSamples:,:]


# In[160]:


valInputData = []
valOutputData = []
trainInputData = []
trainOutputData = []

for i in np.arange(len(shuffledInputTrainingData)):
    valInputData.append(shuffledInputTrainingData[i][:numValidationSamples,:])
    trainInputData.append(shuffledInputTrainingData[i][numValidationSamples:,:])
    # valInputData[i] = shuffledInputTrainingData[i][:numValidationSamples,:]
    # trainInputData[i] = shuffledInputTrainingData[i][numValidationSamples:,:]

for i in np.arange(len(shuffledOutputTrainingData)):
    valOutputData.append(shuffledOutputTrainingData[i][:numValidationSamples,:])
    trainOutputData.append(shuffledOutputTrainingData[i][numValidationSamples:,:])
    # valOutputData[i] = shuffledOutputTrainingData[i][:numValidationSamples,:]
    # trainOutputData[i] = shuffledInputTrainingData[i][numValidationSamples:,:]
    
print('Validation Input Data (stored in list valInputData).\n')
print('Shape: ' + str(np.shape(valInputData)) + '\n')

print('Training Input Data (stored in list trainInputData).\n')
print('Shape: ' + str(np.shape(trainInputData)) + '\n')

print('Validation Output Data (stored in list valOutputData).\n')
print('Shape: ' + str(np.shape(valOutputData)) + '\n')

print('Training Output Data (stored in list trainOutputData).\n')
print('Shape: ' + str(np.shape(trainOutputData)) + '\n')


# In[161]:


#Initialize hyperparamters for Single Fidelity Model

layer1size = 32 
layer2size = 64
numOutputData = 3
rate = 4.0e-3 #From Deep Learning w/ Python (Chollet)
# rate = 5.0e-2 # used by Dr. Reasor
reg = 1.0e-6
numEpochs = 5000
myBatchSize = None
validSplit = 0.2 # 20% of our data will be witheld for validation


# In[162]:


numCases = str(len(WallTemp_Scaled[:,0]))
inputArraySize = str(WallTemp_Scaled.shape)
outputArraySize = str(qw_Scaled.shape)
numInput = str(len(inputVarNames))
numOutput = str(len(outputVarNames))
singleOutput = str(len(qw_Scaled[0,:]))

print('We are using %s cases. Meaning each input array is %s, each output array is %s \n' % 
      (numCases, inputArraySize, outputArraySize))
print('Note: The network, as depicted below, will take data from one case at a time. Therefore,'
     'the input data will be %s 1x1 arrays, and the output will be %s 1x%s arrays \n \n \n' % (numInput, numOutput, singleOutput ))
RANS_NN_val = None #sometimes remnants of previously trained models can hang around, it's best to clear the variable first 
RANS_NN_val = build_model_single_fidelity(WallTemp_Scaled, qw_Scaled, numOutputData, layer1size, layer2size, rate, reg)
tf.keras.utils.plot_model(RANS_NN_val,show_shapes=True)


# In[163]:


# RANS_NN_epochs = None
# RANS_NN_history = None

# RANS_NN_epochs, RANS_NN_history = train_model_all_fidelity(
#     RANS_NN, inputTrainingData, outputTrainingData, numEpochs, myBatchSize, validSplit)

RANS_NN_epochs_val = None
RANS_NN_history_val = None

RANS_NN_epochs_val, RANS_NN_history_val = train_model_all_fidelity(
    RANS_NN_val, inputTrainingData, outputTrainingData, numEpochs, myBatchSize, validSplit)


# In[ ]:


train_mse_qw = RANS_NN_history_val["WallHeatTransfer_mean_squared_error"]
train_mse_P = RANS_NN_history_val["WallPressure_mean_squared_error"]
val_mse_qw = RANS_NN_history_val["val_WallHeatTransfer_mean_squared_error"]
val_mse_P = RANS_NN_history_val["val_WallPressure_mean_squared_error"]

# Input the desired epoch range

epochRangeBegin = 0 
epochRangeEnd = 5000

plt.plot(range(1,len(train_mse_qw[epochRangeBegin:epochRangeEnd]) + 1),
         train_mse_qw[epochRangeBegin:epochRangeEnd],
         label="Heat Transfer Train MSE",linestyle="-", color="k")

plt.plot(range(1,len(val_mse_qw[epochRangeBegin:epochRangeEnd]) + 1),
         val_mse_qw[epochRangeBegin:epochRangeEnd],
         label="Heat Transfer Val MSE",linestyle="-.", color="m")

plt.plot(range(1,len(train_mse_P[epochRangeBegin:epochRangeEnd]) + 1),
         train_mse_P[epochRangeBegin:epochRangeEnd],
         label="Pressure Train MSE",linestyle="-", color="b")

plt.plot(range(1,len(val_mse_P[epochRangeBegin:epochRangeEnd]) + 1),
         val_mse_P[epochRangeBegin:epochRangeEnd],
         label="Pressure Val MSE",linestyle="-.", color="r")

plt.title("Single Fidelity Neural Network Predicition Performance ")
plt.legend(loc=0)
plt.grid()


# In[ ]:


train_mse_qw = RANS_NN_history_val["WallHeatTransfer_mean_squared_error"]
train_mse_P = RANS_NN_history_val["WallPressure_mean_squared_error"]
val_mse_qw = RANS_NN_history_val["val_WallHeatTransfer_mean_squared_error"]
val_mse_P = RANS_NN_history_val["val_WallPressure_mean_squared_error"]

# Input the desired epoch range

epochRangeBegin = 1000 
epochRangeEnd = 5000

plt.plot(range(1,len(train_mse_qw[epochRangeBegin:epochRangeEnd]) + 1),
         train_mse_qw[epochRangeBegin:epochRangeEnd],
         label="Heat Transfer Train MSE",linestyle="-", color="k")

plt.plot(range(1,len(val_mse_qw[epochRangeBegin:epochRangeEnd]) + 1),
         val_mse_qw[epochRangeBegin:epochRangeEnd],
         label="Heat Transfer Val MSE",linestyle="-.", color="m")

plt.plot(range(1,len(train_mse_P[epochRangeBegin:epochRangeEnd]) + 1),
         train_mse_P[epochRangeBegin:epochRangeEnd],
         label="Pressure Train MSE",linestyle="-", color="b")

plt.plot(range(1,len(val_mse_P[epochRangeBegin:epochRangeEnd]) + 1),
         val_mse_P[epochRangeBegin:epochRangeEnd],
         label="Pressure Val MSE",linestyle="-.", color="r")

plt.title("Single Fidelity Neural Network Predicition Performance ")
plt.legend(loc=0)
plt.grid()


# # Multi-Fidelity Modeling

# ## MF Kriging

# In[ ]:


RANS_krig_prediction = RANS_krig.predict(inputDataConcat)


# ## MF Neural Network

# ### Choose Desired Action 

# In[150]:


#Would you like to train a new model? 
trainMultiFidelityNN = True

#Would you like to save your model? 
saveMultiFidelityNN = True

#Would you like to load a previously trained model ?
loadMultiFidelityNN = True

#If you have trained a model in this same workspace as well as loaded a model, would you like to overwrite with your loaded model? 
#If you have not trained a model, but loaded a model, then leave this "True"
if not trainMultiFidelityNN and loadMultiFidelityNN: 
    overwriteMultiFidelityNN = True


# In[ ]:





# ### Assemble Data, Build Model

# ### Flatten Training Data
# Our training data are currently stored in 3D tensors. We need to flatten them to get them to play nice with the neural network and Kriging. 

# In[142]:


# yt_lf_uni1_in = NN_lf_uni1.predict(xt_HF_uni_bench1)
# mf_uni1_input_data = np.concatenate((xt_HF_uni_bench1,yt_lf_uni1_in),
#                                     axis = 1)
# mf_uni1_output_data = yt_hf1_uni_scaled

# ref_size_uni1 = int(len(mf_uni1_input_data))

# NN_mf_uni1 = build_model_multi_fidelity(mf_uni1_input_data, 
#                                         mf_uni1_output_data, ref_size_uni1,
#                                         rate, reg)

# tf.keras.utils.plot_model(NN_mf_uni3,show_shapes=True)

# Input Data (stored in list inputTrainingData):
# input data names: 
#     WallTemp_Scaled
#     Density_Scaled
#     Temperature_Scaled
#     Velocity_Scaled

# output data names: 
#     p_lowFidelity_SE
#     q_lowFidelity

# LFinputDataConcat = np.concatenate((WallTemp_Scaled,Density_Scaled,Temperature_Scaled,Velocity_Scaled),axis=1)
# LFoutputDataConcat = np.concatenate((q_lowFidelity_Scaled,p_lowFidelity_SE_Scaled),axis=1)

HFinputTrainingData = inputTrainingData #for now, let's change this later to see how the model performs

lowFidelityNN_input = LF_NN.predict(HFinputTrainingData)

flattenedLowFidelityNN_input = np.concatenate((lowFidelityNN_input[0],lowFidelityNN_input[1]),
                                    axis = 1)

HFinputTrainingDataNumpy = np.array(HFinputTrainingData).reshape(4,400)

multiFidelityNN_input = np.concatenate((HFinputTrainingDataNumpy.T,flattenedLowFidelityNN_input),
                                    axis = 1)


# In[143]:


outputTrainingData = []
outputTrainingNames = []

print('\nOutput Data (stored in list outputTrainingData):\n')
for i, name in enumerate(outputVarNames):
    ScalerName = name + '_OutputScaler'
    ScaledName = name + '_Scaled'
    OutputDataName = name
    locals()[ScalerName] = None
    locals()[ScalerName] = preprocessing.StandardScaler()
    locals()[ScaledName] = locals()[ScalerName].fit_transform(globals()[OutputDataName])
    outputTrainingData.append(locals()[ScaledName])
    outputTrainingNames.append(ScaledName)
    max_element = str(round(np.max(locals()[ScaledName]),2))
    min_element = str(round(np.min(locals()[ScaledName]),2))
    print(name + ' has been scaled! It is called ' +
          ScaledName + '. Min:' + min_element + '. Max:' + max_element)

multiFidelityNN_output = outputTrainingData


# In[147]:


multiFidelityRefSize = int(len(multiFidelityNN_input))

MF_NN = None
MF_NN = build_model_multi_fidelity(
    input_data = multiFidelityNN_input,
    output_data = qw_Scaled,
    ref_size = multiFidelityRefSize,
    rate = rate,
    reg = reg
)

tf.keras.utils.plot_model(MF_NN,show_shapes=True)


# ### Train Model

# In[149]:


#Initialize hyperparamters for Single Fidelity Model
if trainNN: 
    rate = 1.0e-3 #From Deep Learning w/ Python (Chollet)
    # rate = 5.0e-2 # used by Dr. Reasor
    reg = 1.0e-6
    numEpochs = 1000
    myBatchSize = None
    validSplit = None

    MF_NN_epochs, MF_NN_history = train_model_all_fidelity(
        model = MF_NN, 
        input_data = multiFidelityNN_input,
        output_data = multiFidelityNN_output,
        numEpochs = numEpochs,
        myBatchSize = myBatchSize,
        validSplit = validSplit
    )


# ### Save Model

# In[151]:


if saveMultiFidelityNN:

    # The try/except syntax catches the error if we've already created the directory. 
    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    kerasFolderName = 'MF_NN_'
    dt = str(datetime.date.today())
    kerasFolderName += dt

    try: 
        os.chdir(path)
        mkdirPath = os.path.join(path, kerasFolderName)
        os.mkdir(mkdirPath)
    except: 
        pass 

    # We won't be saving the NN in the same way that we save the GP model. Keras likes to save all kind of other stuff, with no intuitive way to change the file names.
    # For version control purposes (really, to make sure we're not overwriting old models), we'll create a new folder for each saved model with the date included. 

    path = "/home/tyty/Desktop/CFD_Files/MFmodelTestData"
    path += '/' + kerasFolderName
    os.chdir(path)

    model = MF_NN
    model.save(path)

    epochsDict = MF_NN_epochs
    historyDict = MF_NN_history

    epochsFilename = 'MF_NN_epochs'
    historyFilename = 'MF_NN_history'
    dt = str(datetime.date.today())
    ext = '.pkl'
    epochsFilename += '_' + dt + ext
    historyFilename += '_' + dt + ext
    pickle.dump(epochsDict, open(epochsFilename, 'wb'))
    pickle.dump(historyDict, open(historyFilename, 'wb'))


# In[ ]:


# build this one so you just type the date and it populates all the strings/pkls/paths, etc. 

if loadMultiFidelityNN: 
    # Loading the NN is a bit easier-- but you'll need to specify the path. An example path is included already, which will need to be edited if you wish to load a different model. 
    path = '/home/tyty/Desktop/CFD_Files/MFmodelTestData/MF_NN_2022-03-20' # specify desired model 
    os.chdir(path)
    loadedModelName = 'RANS_NN_loaded'

    locals()[loadedModelName] = keras.models.load_model(path)

### Load History and Epochs

if loadHistoryandEpochs:
    # some time later... load the model from disk

    # path variable should already be defined from when you loaded the NN model. If not, uncomment the 2 lines below. 
    # path = '/home/tyty/Desktop/CFD_Files/MFmodelTestData/RANS_NN_2022-02-11'
    # os.chdir(path)

    epochsFilename = 'RANS_NN_epochs_2022-02-26.pkl'
    historyFilename = 'RANS_NN_history_2022-02-26.pkl'

    desiredLoadedEpochsName = 'RANS_NN_epochs_loaded'
    desiredLoadedHistoryName = 'RANS_NN_history_loaded'

    locals()[desiredLoadedEpochsName] = pickle.load(open(epochsFilename, 'rb'))
    locals()[desiredLoadedHistoryName] = pickle.load(open(historyFilename, 'rb'))


# ### Generate Predicitions

# In[ ]:


y_lf_nn_uni1 = NN_lf_uni1.predict(x)

mf_predict_input_uni1 = np.concatenate((x,y_lf_nn_uni1),axis=1)

y_mf_nn_uni1_predicted = NN_mf_uni1.predict(mf_predict_input_uni1)

y_mf_nn_uni1_predicted = hfScalerUni1.inverse_transform(y_mf_nn_uni1_predicted)


# ### Analyze Predictions (One to One Comparison)

# In[ ]:


# plt.rcParams["figure.figsize"] = (10,10)
# fig, axs = plt.subplots(2, 2)
# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# #Reference Functions, Benchmark 1
# axs[0,0].plot(x,hf_function_bench1(x), label="True HF Function",linestyle="-", color="k")
# axs[0,0].plot(x,lf_function_bench1(x), label="True LF Function",linestyle="--", color="k")
# #Predicted Values 
# axs[0,0].plot(x,y_mf_nn_lhs1_predicted, label="MF NN LHS Predicted",linestyle="-.", color="b")
# axs[0,0].plot(x,y_mf_nn_uni1_predicted, label="MF NN Uni Predicted",linestyle=":", color="r")
# axs[0,0].set_title("Benchmark Function 1")
# axs[0,0].legend(loc=0)
# axs[0,0].grid()


# ### Analyze MSE in MF Neural Networks

# In[ ]:


# ###### Low Fidelity Mean Squared Error
# mse_history_mfUni1 = NN_mf_uni1_history["mean_squared_error"]
# mse_history_mfLHS1 = NN_mf_lhs1_history["mean_squared_error"]
# mse_history_mfUni2 = NN_mf_uni2_history["mean_squared_error"]
# mse_history_mfLHS2 = NN_mf_lhs2_history["mean_squared_error"]
# mse_history_mfUni3 = NN_mf_uni3_history["mean_squared_error"]
# mse_history_mfLHS3 = NN_mf_lhs3_history["mean_squared_error"]

# # plt.plot(range(1,len(mse_history[1000:]) + 1),mse_history[1000:])
# # plt.xlabel("Epochs")
# # plt.ylabel("Mean Squared Error")
# # plt.show()

# plt.rcParams["figure.figsize"] = (10,10)
# fig, axs = plt.subplots(2, 2)
# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# #Low Fidelity NN MSE, Benchmark 1
# axs[0,0].plot(range(1,len(mse_history_mfUni1[:]) + 1),mse_history_mfUni1[:],
#               label="MF Uniform MSE",linestyle="-", color="k")
# axs[0,0].plot(range(1,len(mse_history_mfLHS1[:]) + 1),mse_history_mfLHS1[:],
#               label="MF LHS MSE",linestyle="-.", color="b")
# axs[0,0].set_title("Benchmark Function 1")
# axs[0,0].legend(loc=0)
# axs[0,0].grid()

