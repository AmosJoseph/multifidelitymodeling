# %% [markdown]
# # Multi-Fidelity Surrogate Model Comparison - Kriging vs. Deep Neural Net (DNN) 
# This code seeks to compare the data compression, accuracy, portability, and evaluation time for two different types of surrogate modeling techniques: Kriging and Deep Neural Nets (DNN). First, we will build single fidelity models based on the RANS data (our high fidelity data set), then we will build single fidelity models based on local methods (low fidelity data set). Finally, we will build multi-fidelity models combining the data from both models. 
# 
# Our goal is to beat the performance of the single fidelity model, and also potentially explore how much of the high fidelity data is needed--maybe we can match the performace of the single fidelity model, but with significantly less data. 

# %% [markdown]
# # Make Decisions About What You'd Like To Do
# In the below cell, make some choices about where you're executing this program. 

# %%
# Choose your location from the list below. If you are somewhere not listed below, use "other"
location = 1

# personalComputer = 1
# googleColab = 2
# osc = 3
# bolz = 4
# other = 5

## Are you shipping this code to OSC or DoDHPC? 

superComputerTrain = False

## What would you like to do?

# Low Fidelity Models 
LFKrigLoad = False
LFKrigTrain = True
LFKrigSave = True

LFNNLoad = True
LFNNTrain = True
LFNNSave = True

# High Fidelity Models 
HFKrigLoad = False
HFKrigTrain = True
HFKrigSave = True

HFNNLoad = False
HFNNTrain = True
HFNNSave = True

# Multi Fidelity Models 
MFKrigLoad = False
MFKrigTrain = True
MFKrigSave = True

MFNNLoad = False
MFNNTrain = True
MFNNSave = True

# %% [markdown]
# # Progress Notes
# 
# To do: 
# - remove outlier case from the data entirely
# - windowing code cell has bug where if the data is already windowed it gets unhappy. put a try/except or an if statement in there somewhere. 
# - remove all of the x_cc_windowed 
# 
# Long term to do:
# - parametric study (how many RANS cases to keep)
# - hyperparametric study , both kernels for Kriging and HPs for Keras (hyperparameter tuner)
# - try different NN architectures (CNNs, GAN, etc.) (https://www.tensorflow.org/api_docs/python/tf/keras/layers)

# %% [markdown]
# # Import Block

# %%
# %pip install scipy --upgrade
# %pip install gputil
# %pip install tabulate

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'# if you wish to supress TensorFlow debug info, set to 3
import code
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6); plt.rcParams['axes.grid'] = True; plt.rcParams['figure.dpi'] = 300; plt.rcParams['axes.labelsize'] = 'xx-large';
if 
%matplotlib inline
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
import scipy
import GPUtil


from tqdm.keras import TqdmCallback
from sklearn import preprocessing
from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import qmc



# %% [markdown]
# # GPU Details

# %%
import GPUtil
from tabulate import tabulate
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

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
if location == 1: #personalComputer
    path = r'c:\\Users\\tyler\\Desktop\\oscDataPackage'

if location == 2: #googleColab
    # from google.colab import drive
    # drive.mount('/content/drive')
    path = r'/content/drive/MyDrive/Colab Notebooks'

if location == 3: #osc 
    path = r"xxxxxxxx"

if location == 4: #bolz
    path = r"/home/tyty/Desktop/CFD_Files/AFRL_Scholars/1st_Round_Ls"

if location == 5: #other
    path = r"customPath"

os.chdir(path)
print('Current location: ', os.getcwd())
 

# %% [markdown]
# # Functions Block 

# %%
def moveToWorkDir(location):
    personalComputer = 1
    googleColab = 2
    osc = 3
    bolz = 4
    other = 5

    if location == 1: #personalComputer
        path = r'c:\\Users\\tyler\\Desktop\\oscDataPackage'

    if location == 2: #googleColab
        # from google.colab import drive
        # drive.mount('/content/drive')
        path = r'/content/drive/MyDrive/Colab Notebooks'

    if location == 3: #osc 
        path = r"xxxxxxxx"

    if location == 4: #bolz
        path = r"/home/tyty/Desktop/CFD_Files/AFRL_Scholars/1st_Round_Ls"

    if location == 5: #other
        path = r"customPath"

    os.chdir(path)
    print('Current location: ', os.getcwd())

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
        b[i+1] = b[i] - ((2*(1/np.tan(b[i]))*(M**2*(np.sin(b[i]))**2-1))/(M**2*(gamma + np.cos(2*b[i]))+2)-np.tan(theta)) \
        / ((4*M**2*np.sin(2*b[i])*(1/np.tan(b[i]))*(M**2*np.sin(b[i])**2-1))/((M**2*(np.cos(2*b[i])+gamma)+2)**2)
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

#Building the model

def build_model_parameterized(
    input_data, output_data, layerSizeList, rate, regType, regValue,
    hiddenLayerActivation, outputLayerActivation, kernelInitializer,
    optimizer, loss):
    """

    :param input_data: input parameters/features
    :param output_data: outputs the NN is fitting
    :param layerSizeList: list of all the layer sizes. Number of values indicates number of layers
    :param rate: learning rate
    :param reg: L2 regularization value to drop weights
    :return:
    """
    inputlayershape = int(len(input_data[0,:]))
    outputlayershape = int(len(output_data[0,:]))

    inputs = tf.keras.Input(shape=(inputlayershape,))
    
    x = tf.keras.layers.Dense(
        layerSizeList[0],
        activation=hiddenLayerActivation,
        kernel_regularizer=regType(regValue),
        kernel_initializer = kernelInitializer
        )(inputs)

    for layerSize in layerSizeList[1:]:
        x = tf.keras.layers.Dense(
            layerSize,activation=hiddenLayerActivation,
            kernel_regularizer=regType(regValue),
            kernel_initializer = kernelInitializer
            )(x)

    outputs = tf.keras.layers.Dense(
        outputlayershape,
        activation = outputLayerActivation,
        kernel_regularizer = regType(regValue),
        kernel_initializer = kernelInitializer,
        name = 'outputlayer'
        )(x)
    
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer=optimizer(learning_rate=rate),
             loss = loss,
             metrics = [tf.keras.metrics.MeanSquaredError(),
                       tf.keras.metrics.RootMeanSquaredError(),])
                       # "mae"])
    try:
        tf.keras.utils.plot_model(model,show_shapes=True)
    except:
        print('Can\'t show the \'plot model\' thing')
    return model
#training the model 


def train_model_all_fidelity(model, input_data, output_data, numEpochs, myBatchSize, validData, callbacks_list):
    
    history = model.fit(x=input_data,
                       y=output_data,
                       batch_size=myBatchSize,
                       epochs=numEpochs,
                       callbacks = callbacks_list,
                       verbose=False,
                       shuffle=False,
                       validation_data=validData,
                       use_multiprocessing=True)
    epochs = history.epoch
    return epochs, history.history
    
def normalizedRootMeanSquaredError(truth,prediction):
    rmse = mean_squared_error(truth, prediction, squared=False)
    ybar = truth.max()-truth.min()
    nrmse = rmse/ybar
    return nrmse

def errorMetrics(truth,prediction,fidelity,model, variable):
    rmse = mean_squared_error(truth, prediction, squared=False)
    ybar = truth.max()-truth.min()
    nrmse = round(rmse/ybar,5)
    R2 = round(r2_score(truth, prediction),4)
    print(fidelity + ' ' + model + ' ' + variable + ' ' + 'R2: ' + str(R2) )
    print(fidelity + ' ' + model + ' ' + variable + ' ' + 'NRMSE: ' + str(nrmse * 100) + '%\n' )
    return nrmse, R2


def oneToOneVisualizationPlotAllData(
    case, qw_test_predict,p_test_predict, qw_test_truth, p_test_truth, M_inf_test, method):
    plt.rcParams["figure.figsize"] = (15,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=1.5, h_pad=.5)
    fig.patch.set_facecolor('white')

    elbowLocation = 2.35
    case = case
    cm = plt.cm.get_cmap('cool')
    zmax = 2.5
    z = np.arange(0,zmax, zmax/qw_test_predict[case,:].shape[0])
    #plot one case only
    labelstr = 'Mach inf: ' + str(M_inf_test[case]) + ',case:' + str(case)
    maxHeatTransfer = qw_test_predict[case,:].max()
    maxPressure = p_test_predict[case,:].max()
    x = qw_test_predict[case,:]/maxHeatTransfer
    y = qw_test_truth[case,:]/maxHeatTransfer
    sc = axs[0].scatter(x, y ,c = z, s=80, label = labelstr, 
                     cmap=cm,edgecolors='none',vmin=0,vmax=2.5 )
    cbar = fig.colorbar(sc,ax = axs[0])
    cbar.ax.set_title("x-location (meters)")
    cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w')
    axs[0].plot([0, 1], [0, 1], color = 'k')

    caseNRMSE = str(round(100*normalizedRootMeanSquaredError(qw_test_truth[case,:],qw_test_predict[case,:]),4))
    caseR2 =  str(round(r2_score(qw_test_truth[case,:], qw_test_predict[case,:]),4))
    plotTextBox = 'R2: ' + caseR2 + '\n' + 'NRMSE: ' + caseNRMSE + '%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.05, 0.85, plotTextBox, transform=axs[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0].set_title("Heat Transfer Predicitions vs Actual")
    axs[0].grid()
    axs[0].set_ylabel("True Value")
    axs[0].set_xlabel("Predicted Heat Transfer")

    #############################################

    axs1label = method + ' Prediction'
    sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[1].plot(x_cc_sorted[0,idxWindowStart:], qw_test_truth[case,:]/maxHeatTransfer, color='firebrick', 
                linestyle='solid', linewidth=4, label='RANS CFD Output (Truth Data)')

    axs[1].plot(x_cc_sorted[0,idxWindowStart:], qw_test_predict[case,:]/maxHeatTransfer, 
                color='black', linestyle='-.', linewidth=2, label=axs1label)
    axs[1].set_title("Predicted Heat Transfer",fontsize='x-large')
    axs[1].set_ylabel("qw / qw_max", fontsize='x-large')
    axs[1].set_xlabel('x (meters)')

    axs[1].legend(fontsize='x-large')

def oneToOnePlotTool(method, desiredNumCasesForPlot, X_test, qw_prediction, qw_truth, p_prediction, p_truth):
    totalCases = X_test.shape[0]
    casePlotRange= np.arange(0,totalCases,int((totalCases/desiredNumCasesForPlot)))

    plt.rcParams["figure.figsize"] = (10,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
    fig.patch.set_facecolor('white')

    for case in casePlotRange:
        labelstr = 'Case: ' + str(case)
        maxHeatTransfer = max(qw_prediction[case,:].max(),qw_truth[case,:].max())
        maxPressure = max(p_prediction[case,:].max(),p_truth[case,:].max())
        axs[0].scatter(qw_prediction[case,:]/maxHeatTransfer,qw_truth[case,:]/maxHeatTransfer, s=1, label = labelstr )
        axs[1].scatter(p_prediction[case,:]/maxPressure,p_truth[case,:]/maxPressure, s=1, label = labelstr)

    qwCaseNRMSE = str(round(100*normalizedRootMeanSquaredError(qw_truth,qw_prediction),4))
    pCaseNRMSE = str(round(100*normalizedRootMeanSquaredError(p_truth,p_prediction),4))
    qwCaseR2 =  str(round(r2_score(qw_truth, qw_prediction),4))
    pCaseR2 =  str(round(r2_score(p_truth, p_prediction),4))
    qwPlotTextBox = 'R2: ' + qwCaseR2 + '\n' + 'NRMSE: ' + qwCaseNRMSE + '%'
    pPlotTextBox = 'R2: ' + pCaseR2 + '\n' + 'NRMSE: ' + pCaseNRMSE + '%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.05, 0.85, qwPlotTextBox, transform=axs[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1].text(0.05, 0.85, pPlotTextBox, transform=axs[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    axs[0].plot([0, 1], [0, 1], color = 'grey', zorder = 0)
    axs[1].plot([0, 1], [0, 1], color = 'grey', zorder = 0)

    axs0title = method + " Heat Transfer Predicitions vs Actual"
    axs1title = method + " Pressure Predictions vs. Actual"
    axs[0].set_title(axs0title)
    axs[1].set_title(axs1title)
    axs[0].grid()
    axs[1].grid()
    axs[0].set_ylabel("True Value")
    axs[0].set_xlabel("Predicted Heat Transfer")
    axs[1].set_xlabel("Predicted Pressure")

def plotPressureHeatTransferSideBySide(case, qw_test_predict,p_test_predict, qw_test_truth, p_test_truth, method):
    sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    plt.rcParams["figure.figsize"] = (15,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.5)
    fig.patch.set_facecolor('white')

    maxHeatTransfer = max(qw_test_predict[case,:].max(),qw_test_truth[case,:].max())
    maxPressure = max(p_test_predict[case,:].max(),p_test_truth[case,:].max())

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw_test_truth[case,:]/maxHeatTransfer, color='firebrick', linestyle='solid', linewidth=4, label='Truth Data')
    axs[0].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], qw_test_predict[case,::sliceVal]/maxHeatTransfer, c='white',
                zorder=3,edgecolors='black', marker='D', s=70, label=method + ' Prediction')
    axs[0].set_title("Predicted Heat Transfer",fontsize='x-large')
    axs[0].set_ylabel("qw / qw_max", fontsize='x-large')

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_test_truth[case,:]/maxPressure, color='black', linestyle='solid', linewidth=4, label='Truth Data')
    axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], p_test_predict[case,::sliceVal]/maxPressure, c='white',
                zorder=3,edgecolors='black', marker='D', s=70, label=method + ' Prediction')
    axs[1].set_title("Predicted Pressure", fontsize='x-large')
    axs[1].set_ylabel("P/P_max", fontsize='x-large')

    for i in np.arange(0,len(axs)):
        # axs[i].grid()
        axs[i].legend(fontsize='x-large')
        axs[i].set_xlabel('x (meters)',fontsize='x-large')
        # axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
        #     verticalalignment='top', bbox=props)
        
def plotTrainAndTestLoss(historyDict,mseNames,epochRangeBegin,epochRangeEnd,colorList, fidelityLevel):
    numValMSECount = sum(1 for string in mseNames if string.find('val') != -1)
    valMSEArray = np.zeros(((valCount,epochRangeEnd)))
    minMSEArray = np.empty((valCount,1))
    count = 0

    for color,mse in enumerate(mseNames): 
        plt.semilogy(range(1,len(historyDict[mse][epochRangeBegin:epochRangeEnd]) + 1),
         historyDict[mse][epochRangeBegin:epochRangeEnd],
         label=mse,linestyle="-", color=colorList[color])
        if mse.find('val') != -1: 
            minMSEEpochNum = np.argmin(historyDict[mse])
            plt.axvline(x = minMSEEpochNum, linestyle='dashdot', label=mse + ' minimum', color=colorList[color])
            minMSEArray[count] = minMSEEpochNum
            valMSEArray[count,:] = historyDict[mse]
            print('Minimum validation MSE for ' + mse + ' is at epoch number ' + str(minMSEEpochNum) )
            count += 1
    
    summedValMSEArray = valMSEArray.sum(axis=0)

    plt.title(fidelityLevel + " Neural Network Loss")
    plt.legend(loc=0)
    plt.grid()
    # print('Average min epoch number: ' + str(np.mean(minMSEArray)))
    # print('Epoch number, minimum, all val error added: ' + str(np.argmin(summedValMSEArray)))

def plotAverageDistributions(qw_test_predict,p_test_predict, qw_test_truth, p_test_truth, method):
    totalCases = len(qw_test_truth[:,0])
    sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    plt.rcParams["figure.figsize"] = (15,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.5)
    fig.patch.set_facecolor('white')

    mean_qw_test_predict = np.mean(qw_test_predict, axis=0)
    mean_p_test_predict = np.mean(p_test_predict, axis=0)

    mean_qw_test_truth = np.mean(qw_test_truth, axis=0)
    mean_p_test_truth = np.mean(p_test_truth, axis=0)

    maxHeatTransfer = max(mean_qw_test_predict.max(), mean_qw_test_truth.max())
    maxPressure = max(mean_p_test_predict.max(),mean_p_test_truth.max())

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[0].plot(x_cc_sorted[0,idxWindowStart:], mean_qw_test_truth/maxHeatTransfer, color='firebrick', linestyle='solid', linewidth=4, label='Average Truth Data Distribution')
    axs[0].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], mean_qw_test_predict[::sliceVal]/maxHeatTransfer, c='white',
                zorder=3,edgecolors='black', marker='D', s=70, label='Average '+ method + ' Prediction')
    axs[0].set_title("Predicted Heat Transfer",fontsize='x-large')
    axs[0].set_ylabel("qw / qw_max", fontsize='x-large')

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[1].plot(x_cc_sorted[0,idxWindowStart:], mean_p_test_truth/maxPressure, color='black', linestyle='solid', linewidth=4, label='Average Truth Data Distribution')
    axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], mean_p_test_predict[::sliceVal]/maxPressure, c='white',
                zorder=3,edgecolors='black', marker='D', s=70, label='Average '+ method + ' Prediction')
    axs[1].set_title("Predicted Pressure", fontsize='x-large')
    axs[1].set_ylabel("P/P_max", fontsize='x-large')

    for i in np.arange(0,len(axs)):
        # axs[i].grid()
        axs[i].legend(fontsize='x-large')
        axs[i].set_xlabel('x (meters)',fontsize='x-large')
        # axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
        #     verticalalignment='top', bbox=props)

def saveNN(fidelityLevel):
    moveToWorkDir(location)
    # The try/except syntax catches the error if we've already created the directory. 
    kerasFolderName = fidelityLevel + '_NN_'
    dt = str(datetime.date.today())
    kerasFolderName += dt

    try: 
        mkdirPath = os.path.join(path, kerasFolderName)
        os.mkdir(mkdirPath)
    except: 
        pass 

    # We won't be saving the NN in the same way that we save the GP model. Keras likes to save all kind of other stuff, with no intuitive way to change the file names.
    # For version control purposes (really, to make sure we're not overwriting old models), we'll create a new folder for each saved model with the date included. 

    kerasPath = path + '/' + kerasFolderName
    os.chdir(kerasPath)

    modelName = fidelityLevel + '_NN'
    model = globals()[modelName]
    model.save(kerasPath)
    
    epochsName = modelName + '_epochs'
    historyName = modelName + '_history'
    epochsDict = globals()[epochsName]
    historyDict = globals()[historyName]

    epochsFilename = epochsName
    historyFilename = historyName
    dt = str(datetime.date.today())
    ext = '.pkl'
    epochsFilename += '_' + dt + ext
    historyFilename += '_' + dt + ext
    pickle.dump(epochsDict, open(epochsFilename, 'wb'))
    pickle.dump(historyDict, open(historyFilename, 'wb'))

    moveToWorkDir(location)

def loadNN(neuralNetFolderName):
    # Loading the NN is a bit easier-- but you'll need to specify the path. An example path is included already, 
    # which will need to be edited if you wish to load a different model. 
    moveToWorkDir(location)
    loadFolderPath = path + '\\\\' + neuralNetFolderName
    os.chdir(loadFolderPath)
    loadedModelName = neuralNetFolderName[0:2] + '_NN'

    globals()[loadedModelName] = keras.models.load_model(loadFolderPath)

### Load History and Epochs
    # some time later... load the model from disk

    # path variable should already be defined from when you loaded the NN model. If not, uncomment the 2 lines below. 
    # path = '/home/tyty/Desktop/CFD_Files/MFmodelTestData/HF_NN_2022-02-11'
    # os.chdir(path)

    desiredLoadedEpochsName = loadedModelName + '_epochs'
    desiredLoadedHistoryName = loadedModelName + '_history'

    epochsFileName = desiredLoadedEpochsName + neuralNetFolderName[5:] + '.pkl'
    historyFileName =  desiredLoadedHistoryName + neuralNetFolderName[5:] + '.pkl'

    globals()[desiredLoadedEpochsName] = pickle.load(open(epochsFilename, 'rb'))
    globals()[desiredLoadedHistoryName] = pickle.load(open(historyFilename, 'rb'))
    try:
        tf.keras.utils.plot_model(globals()[loadedModelName],show_shapes=True)
    except:
        print('Can\'t show the \'plot model\' thing')
    print(loadedModelName + ' loaded!!!!')
    moveToWorkDir(location)

def krigTrain(X_train,y_train, fidelityLevel, kernel, n_restarts):
    
    modelName = fidelityLevel + '_krig'
    

    globals()[modelName] = None
    globals()[modelName] = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=n_restarts)
    print(modelName + ' training begin')
    start = time.time()
    globals()[modelName].fit(X_train, y_train)

    end = time.time()
    print(modelName + ' training end')
    print(fidelityLevel + ' Kriging train time: %0.7f seconds' % (end-start) )
    print('Model name: ', modelName)
    print('Original kernel:' , kernel)
    print('Optimized kernel:' , globals()[modelName].kernel_ )
    print('_'*25)

def saveKrig(fidelityLevel):
    modelName = fidelityLevel + '_krig'
    model = globals()[modelName]
    filename = modelName + '_'
    dt = str(datetime.date.today())
    ext = '.sav'
    filename += dt + ext
    pickle.dump(model, open(filename, 'wb'))

def generateInverseTransformedPredictions(X_train,X_test,y_train,y_test,method,fidelityLevel):
    # Need to add validation data eventually
    print('Method: ' + method + ', Fidelity Level: ' + fidelityLevel)
    print('_'*25)
    numCases = X_train.shape[0] + X_test.shape[0]
    
    #shorten variable names a bit
    if method == 'kriging':
        method = 'krig'
    
    modelName = fidelityLevel + '_' + method

    dataSplitNames = [
        '_' + modelName + '_test_predict',
        '_' + modelName + '_train_predict',
        '_' + fidelityLevel + '_test_truth',
        '_' + fidelityLevel + '_train_truth'
        ]
    numDataSplit = int(len(dataSplitNames)/2) # are we splitting into train/test (numDataSplit = 2) or train/test/validation (numDataSplit =3) ? 

    #######################################
    # Make this part a bit more robust --> you may lose accuracy on the prediction speed if you loop this though
    # time the model predicting on all of X, then discard that object (set equal to None)
    # then, in a loop, put the right predictions into the right variable names
    start = time.time()
    globals()[dataSplitNames[0][1:]] = globals()[modelName].predict(X_test)
    globals()[dataSplitNames[1][1:]] = globals()[modelName].predict(X_train)
    end = time.time()

    globals()[dataSplitNames[2][1:]] = y_test
    globals()[dataSplitNames[3][1:]] = y_train
    #######################################

    printCount = 0
    # if method == 'krig':
    for i in np.arange(len(outputVarNames)):
        for j in np.arange(numDataSplit):
            #Input Data Split
            globals()[outputVarNames[j] + dataSplitNames[i]] = np.hsplit(globals()[dataSplitNames[i][1:]],numDataSplit)[j]
            print(dataSplitNames[i][1:] + ' part ' + str(j+1) + '--> ' + outputVarNames[j] + dataSplitNames[i])
            globals()[outputVarNames[j] + dataSplitNames[i+2]] = np.hsplit(globals()[dataSplitNames[i+2][1:]],numDataSplit)[j]
            print(dataSplitNames[i+2][1:] + ' part ' + str(j+1) + '--> ' + outputVarNames[j] + dataSplitNames[i+2])

    print('_'*15)
    print('Total prediction time: %0.3f seconds. Prediction time per case: %0.7f seconds, %0.3f Hz ' \
        % (end-start, ((end-start)/numCases), 1/((end-start)/numCases)) )
    print('_'*15)
########### Inverse Scale ###########
# 'qw_LF_LF_krig_test_predict'
    if fidelityLevel == 'LF':
            for name in LFoutputVarNames:
                ScalerName = name + '_OutputScaler'
                for tail in dataSplitNames:
                    fullName = name[0:-3] + tail
                    globals()[fullName] = globals()[ScalerName].inverse_transform(globals()[fullName])
                    print( fullName + ' has been inverse transformed using ' + ScalerName + '! It is called ' + fullName)
    else:
        for name in outputVarNames:
            ScalerName = name + '_OutputScaler'
            for tail in dataSplitNames:
                fullName = name + tail
                globals()[fullName] = globals()[ScalerName].inverse_transform(globals()[fullName])
                print( fullName + ' has been inverse transformed using ' + ScalerName + '! It is called ' + fullName)

# %%
moveToWorkDir(location)

# %% [markdown]
# # Import data from CFD runs. 
# If you have not run the post-processing scripts housed in the top folder, run those first. The below code assumes that all us3d output has been post-processed

# %%
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

# Import the *output* data, i.e. what the RANS code does with our input file. 
# Take these from the files 

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
    
#     plt.rcParams["figure.figsize"] = (20,5)
#     fig, axs = plt.subplots(1, 2)
#     fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
#     fig.patch.set_facecolor('white')


#     axs[0].scatter(x_cc_sorted[caseWePick,idxWindowStart:], p_sorted[caseWePick,idxWindowStart:],s=30, label = 'Windowed Data')
#     axs[0].scatter(x_cc_sorted[caseWePick,:], p_sorted[caseWePick,:],s=3, label = 'Full Data')
#     axs[0].axvline(x=xWindowStart, linewidth=1, color='k', label = 'Chosen Start of Window')
#     axs[1].scatter(x_cc_sorted[caseWePick,idxWindowStart:], qw_sorted[caseWePick,idxWindowStart:],s=30, label = 'Windowed Data')
#     axs[1].scatter(x_cc_sorted[caseWePick,:], qw_sorted[caseWePick,:],s=3, label = 'Full Data')
#     axs[1].axvline(x=xWindowStart, linewidth=1, color='k', label = 'Chosen Start of Window')

#     axs[0].set_title("Pressure Window Results")
#     axs[0].set_ylabel("Pressure (Pa)")
#     axs[0].grid()
#     axs[1].set_title("Heat Transfer Window Results")
#     axs[1].set_ylabel('Heat Transfer (w/m^2')
#     axs[1].grid()

#     axs[0].set_xlabel("x Distance Along Cone")
#     axs[1].set_xlabel("x Distance Along Cone")
    
#     axs[0].legend()
#     axs[1].legend()
    
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

# %% [markdown]
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

# %% [markdown]
# ### Generate Low-Fidelity Space Sample Points 

# %%
dimensionParameterSpace = 4 # wall temp, free stream temp, free stream density, free stream velocity
numPointsToSample = 4000 
seed = 5 # seed with a number for reproducibility. It will be the same LHS sampling every time the function
        # is run using seed = x. 

lowFidelityInputPoints = latinHypercubeSample(
    dimensionParameterSpace,
    numPointsToSample,
    l_bounds,
    u_bounds,
    seed
)

# %%
if not superComputerTrain:

  p.shape

  caseWePick = 147

  plt.rcParams["figure.figsize"] = (15,5)
  fig, axs = plt.subplots(1, 2)
  fig.tight_layout(pad=1.4, w_pad=2.5, h_pad=.5)
  fig.patch.set_facecolor('white')

  axs[0].plot(x_cc_sorted[caseWePick,idxWindowStart:], p[caseWePick,:], label = 'Pressure')
  axs[1].plot(x_cc_sorted[caseWePick,idxWindowStart:], qw[caseWePick,:]/10000,label = 'Heat Transfer')

  axs[0].set_ylabel("Pressure (Pa)")
  axs[1].set_ylabel('Heat Transfer (w/cm^2')

  axs[0].set_xlabel("x Distance Along Cone")
  axs[1].set_xlabel("x Distance Along Cone")

  axs[0].legend()
  axs[1].legend()

# %% [markdown]
# ### Input Data
# 
# A quick explanation of our models' output data 
# 
# qw = heat transfer at the wall
# 
# p = pressure at the wall
# 
# We won't be training the model using the x,y,z,tau,yplus, T, or the cell center locations. These will be used after the model is trained for analysis purposes. Note that the "T", or wall temperature value, was fixed by the CFD code as the thermal boundary condition. If you take a look inside the output T array, you'll find that it's all the same number as the input wall temperature. 

# %% [markdown]
# # Generate Low Fidelity Data
# Shock Expansion and Eckert's Reference Temp
# 

# %% [markdown]
# ## Shock Expansion

# %%
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
xPressureWindowLeft = 2.353056 # elbow location
xPressureWindowRight = 2.5039961 # end of cone

xPressureWindowMid = 2.4
xSpotLeft = (np.abs(x_cc_windowed[0,:] - xPressureWindowLeft)).argmin()
xSpotRight = (np.abs(x_cc_windowed[0,:] - xPressureWindowRight)).argmin()

meanPressure40DegConeSection = np.median(p[:,xSpotLeft:xSpotRight], axis = 1).reshape(-1,1)
cp_actual2 = (meanPressure40DegConeSection - P_inf2)/ (0.5*rho_inf2*(u_inf2**2))
cp_actual2[389] = None # takes care of that one bad case

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

p_lowFidelity_SE_truncated = np.concatenate((p_SE_7deg, p_SE_40deg), axis=1)
p_lowFidelity_SE_truncated = p_lowFidelity_SE_truncated.T

normalizedHFPressure = p/P03
normalizedLFPressure = p_lowFidelity_SE/P03
normalizedLFPressureNewt = p_lowFidelity_Newt/P03

p_LF = p_lowFidelity_SE

# %% [markdown]
# ### Plotting Pressure Coefficient Results

# %%
if not superComputerTrain:

  caseWePick = random.randint(0,400)

  plt.plot(x_cc_windowed[0,:],normalizedLFPressure[caseWePick,:], label = 'Low Fidelity (SE)', color = 'k')
  plt.plot(x_cc_windowed[0,:],normalizedLFPressureNewt[caseWePick,:], label = 'Low Fidelity (Newtonian)', color = 'b', linestyle='-.')
  plt.scatter(x_cc_sorted[caseWePick,idxWindowStart:], normalizedHFPressure[caseWePick,:],s=5, label = 'Truth Data (RANS)', color='r')
  plt.legend()
  # plt.title('Case Number ' + str(caseWePick))
  plt.xlabel('x (meters)')
  plt.ylabel('$P/P_{03}$')
  plt.grid()

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

  # # TURN THIS INTO A SUBLPLOT, side by side

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

# %% [markdown]
# ## Eckert's Reference Temperature

# %%
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

qw_LF = np.concatenate((q_dot_FrontCone, q_dot_RearCone), axis=1)

# %%
if not superComputerTrain:

  plt.semilogy(x_cc_windowed[0,:],qw_LF[caseWePick,:],label = 'Low Fidelity Heat Transfer')
  plt.semilogy(x_cc_windowed[0,:],qw[caseWePick,:], label = 'Truth (RANS) Heat Transfer')
  plt.grid()
  plt.xlabel('Location Along Wedge Wall, $x$ (meters)')
  plt.ylabel('Heat Transfer Rate ($q_w$)')
  plt.legend()

# %% [markdown]
# # Low Fidelity Data Processing

# %% [markdown]
# ### Data Scaling

# %%
LFoutputVarNames = [
    'qw_LF',
    'p_LF',
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

LFoutputTrainingData = []
LFoutputTrainingNames = []

print('\nOutput Data (stored in list LFoutputTrainingData):\n')
for i, name in enumerate(LFoutputVarNames):
    ScalerName = name + '_OutputScaler'
    ScaledName = name + '_Scaled'
    OutputDataName = name
    locals()[ScalerName] = None
    locals()[ScalerName] = preprocessing.StandardScaler()
    locals()[ScaledName] = locals()[ScalerName].fit_transform(globals()[OutputDataName])
    LFoutputTrainingData.append(locals()[ScaledName])
    LFoutputTrainingNames.append(ScaledName)
    max_element = str(round(np.max(locals()[ScaledName]),2))
    min_element = str(round(np.min(locals()[ScaledName]),2))
    print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

print(str(np.shape(inputTrainingData)))
print(str(np.shape(LFoutputTrainingData)))
print(inputTrainingNames)
print(LFoutputVarNames)
print(LFoutputTrainingNames)

# %% [markdown]
# ### Validation Split 

# %%
# these values are already calculated in the low fidelity data generation section, but repeated here in the event that
# that section of the notebook has not been run yet. 

gamma = 1.4 # perfect gas
R_specific = 287.058

T_inf = inputTemperature
rho_inf = inputDensity
u_inf = inputVelocity
a_inf = np.sqrt(gamma*R_specific*T_inf)
M_inf = u_inf/a_inf

# %%
##### SKLEARN DATA SPLIT 
random_state = 30 # used for reproducibility of data splits

X = np.hstack(inputTrainingData)
y_lf = np.hstack(LFoutputTrainingData)
Y_lf_names = np.hstack(LFoutputTrainingNames)
X_names = np.hstack(inputTrainingNames)

X_train, X_test, y_lf_train, y_lf_test = train_test_split(X, y_lf, test_size=0.20, random_state=random_state)
X_test, X_val, y_lf_test, y_lf_val = train_test_split(X_test, y_lf_test, test_size=0.50, random_state=random_state)
M_inf_train, M_inf_test = train_test_split(M_inf,test_size=0.20,random_state=random_state) # used in plotting
M_inf_test, M_inf_val = train_test_split(M_inf_test,test_size=0.50,random_state=random_state) # used in plotting

print("Low fidelity X_train shape: {}".format(X_train.shape))
print("Low fidelity X_test shape: {}".format(X_test.shape))
print("Low fidelity X_val shape: {}".format(X_val.shape))
print("Low fidelity y_lf_train shape: {}".format(y_lf_train.shape))
print("Low fidelity y_lf_test shape: {}".format(y_lf_test.shape))
print("Low fidelity y_lf_val shape: {}".format(y_lf_val.shape))
print(f"concatenation order: {X_names}")
print(f"concatenation order: {Y_lf_names}")

# %% [markdown]
# # LF Kriging

# %%
# Kernel options include: Compound, Constant, Dot Product, Exponentiation, ExpSineSquared, 
# Product, Rational Quadratic, RBF, Sum, White

# Benchmark Function 1
kernel1 = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(1.0, (1e-8, 1e2)) # Works great for Benchmark Function 1

#Benchmark Function 2
kernel2 = 66.3*RBF(0.1, (1e-8, 1e2)) #seems to work well for discontinuities, so we'll try for Benchmark Function 2

#Benchmark Function 3
kernel3 = kernel2

LFoptimizedKernel = 2.44**2 * RBF(length_scale=2.41)
# Below are the different kernels (and combinations of kernels) that Dr. Reasor used in his code

#kernel = ConstantKernel(1.0) + Matern(length_scale=0.1, nu=3/2) #+ WhiteKernel(noise_level=1)
#kernel = 1.0*Matern(length_scale=0.1, nu=1.5)
#kernel = RationalQuadratic()
#kernel = Matern(length_scale=0.1, nu=2.5)  #
#kernel = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(0.1, (1e-8, 1e2))
#kernel = Cdatetime.dateonstantKernel(1.0, (1e-3, 1e3))*RBF(1.0, (1e-2, 1e2))

# %% [markdown]
# ## Train LF Kriging

# %%
# to get this to converge, we need to change the optimizer in a manual way. possibly through "inheritance," 
# or maybe by copying the source code and modifying the optimizer to increase max iters
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
# https://github.com/scikit-learn/scikit-learn/blob/16625450b/sklearn/gaussian_process/_gpr.py#L24

if LFKrigTrain: 

  krigTrain(
    X_train=X_train,
    y_train=y_lf_train, 
    fidelityLevel='LF',
    kernel=LFoptimizedKernel, 
    n_restarts = 1
    )
    

# %% [markdown]
# ## Save LF Kriging

# %%
if LFKrigSave:
    saveKrig(fidelityLevel='LF')

# %% [markdown]
# ## Load LF Kriging

# %%
if LFKrigLoad:

    filename = 'LF_krig_2022-05-17.sav'
    desiredLoadedModelName = 'LF_krig_loaded'

    locals()[desiredLoadedModelName] = pickle.load(open(filename, 'rb'))
    
    LF_krig = None
    LF_krig = LF_krig_loaded
    LF_krig_loaded = None

# %% [markdown]
# ## Generate Predictions

# %%
generateInverseTransformedPredictions(
    X_train = X_train,
    X_test = X_test,
    y_train = y_lf_train,
    y_test = y_lf_test,
    method = 'kriging',
    fidelityLevel = 'LF'
    )

# %%
oneToOnePlotTool(method='Kriging', 
                 desiredNumCasesForPlot=12, 
                 X_test=X_test, 
                 qw_prediction = qw_LF_krig_test_predict, 
                 qw_truth = qw_LF_test_truth, 
                 p_prediction = p_LF_krig_test_predict, 
                 p_truth = p_LF_test_truth)

# %%
oneToOneVisualizationPlotAllData(case=10, 
    qw_test_predict=qw_LF_krig_test_predict,
    p_test_predict=p_LF_krig_test_predict, 
    qw_test_truth=qw_LF_test_truth, 
    p_test_truth=p_LF_test_truth, 
    M_inf_test=M_inf_test,
    method='Kriging')

# %% [markdown]
# # Low Fidelity Neural Network 

# %%
#Initialize hyperparamters for Single Fidelity Model
if LFNNTrain:
      lowFidelityLayerSizeList = [32, 64] #input the number of neurons per layer. Length of this list indicates number of hidden layers
      learningRate = 1.0e-3 #From Deep Learning w/ Python (Chollet)
      regType = keras.regularizers.L2
      regValue = 1.0e-6
      hiddenLayerActivation = tf.nn.tanh
      outputLayerActivation = tf.nn.leaky_relu
      kernelInitializer = tf.keras.initializers.GlorotUniform()
      optimizer = tf.keras.optimizers.Adamax
      numEpochs = 6000
      myBatchSize = None
      validSplit = None
      loss = 'mse'
      validData = (X_test, y_lf_test)

      callbacks_list = None

      LF_NN = None #sometimes remnants of previously trained models can hang around, it's best 
                  #to clear the variable first 

      LF_NN = build_model_parameterized(
            input_data = X_train, 
            output_data = y_lf_train,
            layerSizeList = lowFidelityLayerSizeList, 
            rate = learningRate, 
            regType = regType, 
            regValue = regValue,
            hiddenLayerActivation = hiddenLayerActivation,
            outputLayerActivation = outputLayerActivation,
            kernelInitializer = kernelInitializer,
            optimizer = optimizer,
            loss = loss)
try:
      tf.keras.utils.plot_model(LF_NN,show_shapes=True)
except:
      print('Cannot show model graph, tf.keras.utils.plot_model not loaded')
LF_NN.summary()

# %%
tf.debugging.set_log_device_placement(True)

if LFNNTrain:

        # LFNNOutputTrainingData = np.hsplit(y_lf_train,2)
        # LFNNOutputTestData = np.hsplit(y_lf_test,2)
        # X_train_NN = np.hsplit(X_train,4)
        # X_test_NN = np.hsplit(X_test,4)
        
        LF_NN_epochs = None
        LF_NN_history = None
        print('GPU training start')
        start = time.time()
        LF_NN_epochs, LF_NN_history = train_model_all_fidelity(
            model = LF_NN, 
            input_data = X_train, 
            output_data = y_lf_train,
            numEpochs = numEpochsLF, 
            myBatchSize = myBatchSize,
            validData = validData,
            callbacks_list= callbacks_list)
        end = time.time()
        print('GPU training complete')
        print(round((end-start),4))



# %%
with tf.device('/CPU:0'):
    if LFNNTrain:

        LFNNOutputTrainingData = np.hsplit(y_lf_train,2)
        LFNNOutputTestData = np.hsplit(y_lf_test,2)
        X_train_NN = np.hsplit(X_train,4)
        X_test_NN = np.hsplit(X_test,4)
        
        LF_NN_epochs = None
        LF_NN_history = None
        print('CPU training start')
        start = time.time()
        LF_NN_epochs, LF_NN_history = train_model_all_fidelity(
            LF_NN, X_train_NN, LFNNOutputTrainingData, 
            numEpochsLF, myBatchSize, validData = (X_test_NN, LFNNOutputTestData))
        end = time.time()
        print('CPU training Complete')
        print(round((end-start),4))


# %%
if LFNNSave:
    saveNN(fidelityLevel='LF')

# %% [markdown]
# ## Load LF NN

# %%
if LFNNLoad: 
    loadNN(
        neuralNetFolderName = 'LF_NN_2022-05-19'
        )

# %% [markdown]
# ## Plot Training MSE for LF NN

# %%
mseNames = ["mean_squared_error",
            'val_mean_squared_error'
            ]
colorList = [ 'k', 'r']

plotTrainAndTestLoss(historyDict = LF_NN_history,
                     mseNames = mseNames,
                     epochRangeBegin = 0,
                     epochRangeEnd = numEpochsLF,
                     colorList = colorList,
                     fidelityLevel = 'Low-Fidelity')

# %% [markdown]
# ## LF NN Predictions

# %%
generateInverseTransformedPredictions(
    X_train = X_train,
    X_test = X_test,
    y_train = y_lf_train,
    y_test = y_lf_test,
    method = 'NN',
    fidelityLevel = 'LF'
    )

# %%
oneToOnePlotTool(method='Kriging', 
                 desiredNumCasesForPlot=12, 
                 X_test=X_test, 
                 qw_prediction = qw_LF_krig_test_predict, 
                 qw_truth = qw_LF_test_truth, 
                 p_prediction = p_LF_krig_test_predict, 
                 p_truth = p_LF_test_truth)

# %%
oneToOneVisualizationPlotAllData(32, 
    qw_LF_NN_test_predict,p_LF_NN_test_predict, qw_LF_test_truth, p_LF_test_truth, M_inf_test, method='NN')

# %% [markdown]
# # Low Fidelity Error Metrics Comparison 

# %%
[LF_krig_NRMSE_pressure, LF_krig_R2_pressure] = errorMetrics(
    truth = p_LF_test_truth,
    prediction = p_LF_krig_test_predict,
    fidelity = 'LF',
    model = 'Kriging',
    variable = 'Pressure')

[LF_krig_NRMSE_heatTransfer, LF_krig_R2_heatTransfer] = errorMetrics(
    truth = qw_LF_test_truth,
    prediction = qw_LF_krig_test_predict,
    fidelity = 'LF',
    model = 'Kriging',
    variable = 'Heat Transfer')

[LF_NN_NRMSE_pressure, LF_NN_R2_pressure] = errorMetrics(
    truth = p_LF_test_truth,
    prediction = p_LF_NN_test_predict,
    fidelity = 'LF',
    model = 'NN',
    variable = 'Pressure')

[LF_NN_NRMSE_heatTransfer, LF_NN_R2_heatTransfer] = errorMetrics(
    truth = qw_LF_test_truth,
    prediction = qw_LF_NN_test_predict,
    fidelity = 'LF',
    model = 'NN',
    variable = 'Heat Transfer')

# %% [markdown]
# # High Fidelity Data Processing 
# 
# ## Data scaling
# using sklearn's ```Standard.Scaler```
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
# 
# ## Validation Split
# Validation Split: ```sklearn.model_selection.train_test_split```
# 
# [scikit-learn: train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

# %%
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

# %%
# these values are already calculated in the low fidelity data generation section, but repeated here in the event that
# that section of the notebook has not been run yet. 

gamma = 1.4 # perfect gas
R_specific = 287.058

T_inf = inputTemperature
rho_inf = inputDensity
u_inf = inputVelocity
a_inf = np.sqrt(gamma*R_specific*T_inf)
M_inf = u_inf/a_inf

# %%
##### SKLEARN DATA SPLIT 
random_state = 30 # used for reproducibility of data splits

X = np.hstack(inputTrainingData)
y = np.hstack(outputTrainingData)
Y_names = np.hstack(outputTrainingNames)
X_names = np.hstack(inputTrainingNames)
originalIdx = np.arange(0,400)

X_train, X_test, y_train, y_test, M_inf_train, M_inf_test, trainIdx, testIdx = train_test_split(
    X, y, M_inf, originalIdx, test_size=0.20, random_state=random_state)

X_test, X_val, y_test, y_val, M_inf_test, M_inf_val, testIdx, valIdx = train_test_split(
    X_test, y_test, M_inf_test, testIdx, test_size=0.50, random_state=random_state)
# M_inf_train, M_inf_test = train_test_split(M_inf,test_size=0.20,random_state=random_state) # used in plotting
# M_inf_test, M_inf_val = train_test_split(M_inf_test,test_size=0.50,random_state=random_state) # used in plotting

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_val shape: {}".format(y_val.shape))
print(f"concatenation order: {X_names}")
print(f"concatenation order: {Y_names}")

# %% [markdown]
# # Kriging (Gaussian Process, or GP)

# %% [markdown]
# ## Gaussian Process Kernels
# What kernels do/how to select kernels:
# 
# [The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/)
# 

# %%
kernel2 = 66.3*RBF(0.1, (1e-9, 1e2)) + WhiteKernel(noise_level=1) #seems to work well for discontinuities on the analytical functions
kernel = 1.0*Matern(length_scale=0.1, nu=1.5)

HFoptimizedKernel = 2.28**2 * RBF(length_scale=3.09) + WhiteKernel(noise_level=0.002)

# Below are the different kernels (and combinations of kernels) that Dr. Reasor used in his code

#kernel = ConstantKernel(1.0) + Matern(length_scale=0.1, nu=3/2) #+ WhiteKernel(noise_level=1)
#kernel = 1.0*Matern(length_scale=0.1, nu=1.5)
#kernel = RationalQuadratic()
#kernel = Matern(length_scale=0.1, nu=2.5)  #
#kernel = ConstantKernel(1.0, (1e-8, 1e2)) * RBF(0.1, (1e-8, 1e2))
#kernel = Cdatetime.dateonstantKernel(1.0, (1e-3, 1e3))*RBF(1.0, (1e-2, 1e2))

# %% [markdown]
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

# %% [markdown]
# $\textit{Some words about multi-output Kriging} $
# 
# Multi-input, multi-output Kriging
# 
# "At the heart of your issue lies something rarely mentioned (or even hinted at) in practice and in relevant tutorials: Gaussian Process regression with multiple outputs is highly non-trivial and still a field of active research. Arguably, scikit-learn cannot really handle the case, despite the fact that it will superficially appear to do so, without issuing at least some relevant warning."
# 
# https://stackoverflow.com/questions/50185399/multiple-output-gaussian-process-regression-in-scikit-learn

# %% [markdown]
# ## Building the GP Model and Fitting the Data

# %%
if HFKrigTrain:
    krigTrain(
        X_train=X_train,
        y_train=y_train, 
        fidelityLevel='HF',
        kernel=kernel2, 
        n_restarts = 2
    )

# %% [markdown]
# ## Saving or loading your models
# 
# The larger your data gets, and the longer the train time gets, the more you may be interested in saving/loading a pre-trained model. 

# %%
## Saving your model
if HFKrigSave:

    saveKrig(fidelityLevel='HF')

# %%
# some time later... load the model from disk
if HFKrigLoad:

    filename = 'HF_krig_2022-04-14.sav'
    desiredLoadedModelName = 'HF_krig_loaded'

    locals()[desiredLoadedModelName] = pickle.load(open(filename, 'rb'))

    HF_krig = None
    HF_krig = RANS_krig_loaded

# %% [markdown]
# ## Inverse Transform

# %%
generateInverseTransformedPredictions(
    X_train = X_train,
    X_test = X_test,
    y_train = y_train,
    y_test = y_test,
    method = 'kriging',
    fidelityLevel = 'HF'
    )

# %% [markdown]
# ## Analyze Our Predictions

# %%
oneToOnePlotTool(method='Kriging', 
                 desiredNumCasesForPlot=12, 
                 X_test=X_test, 
                 qw_prediction = qw_HF_krig_test_predict, 
                 qw_truth = qw_HF_test_truth, 
                 p_prediction = p_HF_krig_test_predict, 
                 p_truth = p_HF_test_truth)

# %% [markdown]
# plt.fill_between(
#     X.ravel(),
#     mean_prediction - 1.96 * std_prediction,
#     mean_prediction + 1.96 * std_prediction,
#     alpha=0.5,
#     label=r"95% confidence interval",
# )
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py

# %%
plotPressureHeatTransferSideBySide(
    case = 11,
    qw_test_predict = qw_HF_krig_test_predict,
    p_test_predict = p_HF_krig_test_predict,
    qw_test_truth = qw_HF_test_truth,
    p_test_truth = p_HF_test_truth,
    method = 'Kriging')

# %% [markdown]
# ## Mean Predicition and Truth

# %%
plotAverageDistributions(
    qw_test_predict = qw_HF_krig_test_predict,
    p_test_predict = p_HF_krig_test_predict, 
    qw_test_truth = qw_HF_test_truth, 
    p_test_truth = p_HF_test_truth, 
    method = 'Kriging')

# %% [markdown]
# ## One to One Plot Visualizations

# %% [markdown]
# https://stackoverflow.com/questions/33275189/how-can-i-make-points-of-a-python-plot-appear-over-time?noredirect=1&lq=1

# %%
oneToOneVisualizationPlotAllData(
    case = 8, 
    qw_test_predict = qw_HF_krig_test_predict,
    p_test_predict = p_HF_krig_test_predict,
    qw_test_truth = qw_HF_test_truth, 
    p_test_truth = p_HF_test_truth, 
    M_inf_test = M_inf_test,
    method = 'HF Kriging')

# %% [markdown]
# # Artificial Neural Network
# Below are the functions that create and train the Neural Network models

# %% [markdown]
# ## Initialize Hyperparameters

# %% [markdown]
# ### NN-specific data processing
# 
# I've built the neural network to intake 4 operating conditions (assembled in a list) and output 2 distributions (also assembled in a list), so we're splitting the data back up again. 

# %% [markdown]
# ## Build Model and Plot Graph

# %%
#Initialize hyperparamters for Single Fidelity Model
if HFNNTrain: 
    highFidelityLayerSizeList = [32, 64] #input the number of neurons per layer. Length of this list indicates number of hidden layers
    learningRate = 1.0e-3 #From Deep Learning w/ Python (Chollet)
    regType = keras.regularizers.L2
    regValue = 1.0e-6
    hiddenLayerActivation = tf.nn.tanh
    outputLayerActivation = tf.nn.leaky_relu
    kernelInitializer = tf.keras.initializers.GlorotUniform()
    optimizer = tf.keras.optimizers.Adamax
    numEpochs = 4510
    myBatchSize = None
    validSplit = None
    loss = 'mse'
    validData = (X_test, y_test)

    callbacks_list = None

    HF_NN = None #sometimes remnants of previously trained models can hang around, it's best 
                #to clear the variable first 

    HF_NN = build_model_parameterized(
        input_data = X_train, 
        output_data = y_train,
        layerSizeList = highFidelityLayerSizeList, 
        rate = learningRate, 
        regType = regType, 
        regValue = regValue,
        hiddenLayerActivation = hiddenLayerActivation,
        outputLayerActivation = outputLayerActivation,
        kernelInitializer = kernelInitializer,
        optimizer = optimizer,
        loss = loss)
try:
    tf.keras.utils.plot_model(HF_NN,show_shapes=True)
except:
    print('Cannot show model graph, tf.keras.utils.plot_model not loaded')
HF_NN.summary()

# %% [markdown]
# ## Train the Model

# %%
if HFNNTrain: 

    HF_NN_epochs = None
    HF_NN_history = None
    print('HF NN training start')
    start = time.time()
    HF_NN_epochs, HF_NN_history = train_model_all_fidelity(
        model = HF_NN, 
        input_data = X_train, 
        output_data = y_train,
        numEpochs = numEpochs, 
        myBatchSize = myBatchSize,
        validData = validData,
        callbacks_list= callbacks_list)
    end = time.time()
    print('HF NN training complete. Time to train:')
    print(round((end-start),4))

# %% [markdown]
# ### Save NN, History, and Epochs

# %%
if HFNNSave:
    saveNN(fidelityLevel='HF')

# %% [markdown]
# ### Load NN

# %%
if HFNNLoad: 
    loadNN(
        neuralNetFolderName = 'HF_NN_2022-XX-XX'
        )

# %% [markdown]
# ## Summary of Model

# %%
HF_NN.summary()

# %% [markdown]
# ## Analyze Error (MSE)

# %%
mseNames = ["mean_squared_error",
            'val_mean_squared_error'
            ]
colorList = [ 'k', 'r']

plotTrainAndTestLoss(historyDict = HF_NN_history,
                     mseNames = mseNames,
                     epochRangeBegin = 0,
                     epochRangeEnd = numEpochs,
                     colorList = colorList,
                     fidelityLevel = 'High Fidelity')

# %% [markdown]
# ## Generate and Inverse Transform Predictions

# %%
generateInverseTransformedPredictions(
    X_train = X_train,
    X_test = X_test,
    y_train = y_train,
    y_test = y_test,
    method = 'NN',
    fidelityLevel = 'HF'
    )

# %% [markdown]
# ## Analyze Predictions
# For 20 cases, we have plotted the predicted value vs. the actual value. A perfect prediciton would be a straight line with a slope of 1. 

# %%
oneToOnePlotTool(method='Neural Network', 
                 desiredNumCasesForPlot=12, 
                 X_test=X_test, 
                 qw_prediction = qw_HF_NN_test_predict, 
                 qw_truth = qw_HF_test_truth, 
                 p_prediction = p_HF_NN_test_predict, 
                 p_truth = p_HF_test_truth)

oneToOneVisualizationPlotAllData(
    case = 8, 
    qw_test_predict = qw_HF_NN_test_predict,
    p_test_predict = p_HF_NN_test_predict,
    qw_test_truth = qw_HF_test_truth, 
    p_test_truth = p_HF_test_truth, 
    M_inf_test = M_inf_test,
    method = 'HF Neural Network')

plotPressureHeatTransferSideBySide(
    case = 11,
    qw_test_predict = qw_HF_NN_test_predict,
    p_test_predict = p_HF_NN_test_predict,
    qw_test_truth = qw_HF_test_truth,
    p_test_truth = p_HF_test_truth,
    method = 'NN')

# %% [markdown]
# ## Mean Predicition and Truth

# %%
plotAverageDistributions(
    qw_test_predict = qw_HF_NN_test_predict,
    p_test_predict = p_HF_NN_test_predict,
    qw_test_truth = qw_HF_test_truth,
    p_test_truth = p_HF_test_truth,
    method = 'NN')

# %% [markdown]
# # High Fidelity Error Metrics Comparison

# %%
[HF_krig_NRMSE_pressure, HF_krig_R2_pressure] = errorMetrics(
    truth = p_HF_test_truth,
    prediction = p_HF_krig_test_predict,
    fidelity = 'HF',
    model = 'Kriging',
    variable = 'Pressure')

[HF_krig_NRMSE_heatTransfer, HF_krig_R2_heatTransfer] = errorMetrics(
    truth = qw_HF_test_truth,
    prediction = qw_HF_krig_test_predict,
    fidelity = 'HF',
    model = 'Kriging',
    variable = 'Heat Transfer')

[HF_NN_NRMSE_pressure, HF_NN_R2_pressure] = errorMetrics(
    truth = p_HF_test_truth,
    prediction = p_HF_NN_test_predict,
    fidelity = 'HF',
    model = 'NN',
    variable = 'Pressure')

[HF_NN_NRMSE_heatTransfer, HF_NN_R2_heatTransfer] = errorMetrics(
    truth = qw_HF_test_truth,
    prediction = qw_HF_NN_test_predict,
    fidelity = 'HF',
    model = 'NN',
    variable = 'Heat Transfer')

# %% [markdown]
# # Multi-Fidelity Modeling

# %% [markdown]
# ## MF Kriging

# %%
lowFidelityKriging_input = LF_krig.predict(X)
multiFidelityKrig_input = np.concatenate((X,lowFidelityKriging_input),
                                    axis = 1)
multiFidelityKrig_input.shape

##### SKLEARN DATA SPLIT 
random_state = 30 # used for reproducibility of data splits

X_mf = multiFidelityKrig_input
y_mf = np.hstack(outputTrainingData)
Y_mf_names = np.hstack(outputVarNames)
X_names = np.hstack(inputTrainingNames)

X_mf_train, X_mf_test, y_mf_train, y_mf_test, M_inf_train, M_inf_test = train_test_split(
    X_mf, y_mf, M_inf, test_size=0.20, random_state=random_state)

X_mf_test, X_mf_val, y_mf_test, y_mf_val, M_inf_test, M_inf_val = train_test_split(
    X_mf_test, y_mf_test, M_inf_test, test_size=0.50, random_state=random_state)

# M_inf_train, M_inf_test = train_test_split(M_inf,test_size=0.20,random_state=random_state) # used in plotting
# M_inf_test, M_inf_val = train_test_split(M_inf_test,test_size=0.50,random_state=random_state) # used in plotting

print("Multi-fidelity X_train shape: {}".format(X_mf_train.shape))
print("Multi-fidelity X_test shape: {}".format(X_mf_test.shape))
print("Multi-fidelity X_val shape: {}".format(X_mf_val.shape))
print("Multi-fidelity y_lf_train shape: {}".format(y_mf_train.shape))
print("Multi-fidelity y_lf_test shape: {}".format(y_mf_test.shape))
print("Multi-fidelity y_lf_val shape: {}".format(y_mf_val.shape))
print(f"concatenation order: {X_names}")
print(f"concatenation order: {Y_mf_names}")

# %%
if MFKrigTrain:
    
    MFoptimizedKernel =  1.02**2 * RBF(length_scale=9.74)
    
    krigTrain(
        X_train=X_mf_train,
        y_train=y_mf_train, 
        fidelityLevel='MF',
        kernel=MFoptimizedKernel, 
        n_restarts = 1
        )

# %%
if MFKrigSave:
    saveKrig(fidelityLevel='MF')

# %%
if MFKrigLoad:

    filename = 'MF_krig_2022-03-21.sav'
    desiredLoadedModelName = 'MF_krig_loaded'

    locals()[desiredLoadedModelName] = pickle.load(open(filename, 'rb'))
#########################
    MF_krig = None
    MF_krig = MF_krig_loaded

# %% [markdown]
# ### Split and Scale Output

# %%
generateInverseTransformedPredictions(
    X_train = X_mf_train,
    X_test = X_mf_test,
    y_train = y_mf_train,
    y_test = y_mf_test,
    method = 'kriging',
    fidelityLevel = 'MF'
    )

# %%
oneToOnePlotTool(method='Kriging', 
                 desiredNumCasesForPlot=12, 
                 X_test=X_mf_test, 
                 qw_prediction = qw_MF_krig_test_predict, 
                 qw_truth = qw_MF_test_truth, 
                 p_prediction = p_MF_krig_test_predict, 
                 p_truth = p_MF_test_truth)

oneToOneVisualizationPlotAllData(
    case = 7, 
    qw_test_predict = qw_MF_krig_test_predict,
    p_test_predict = p_MF_krig_test_predict,
    qw_test_truth = qw_MF_test_truth, 
    p_test_truth = p_MF_test_truth, 
    M_inf_test = M_inf_test,
    method = 'Kriging')

# %%
[MF_krig_NRMSE_pressure, MF_krig_R2_pressure] = errorMetrics(
    truth = p_MF_test_truth,
    prediction = p_MF_krig_test_predict,
    fidelity = 'MF',
    model = 'Kriging',
    variable = 'Pressure')

[MF_krig_NRMSE_heatTransfer, MF_krig_R2_heatTransfer] = errorMetrics(
    truth = qw_MF_test_truth,
    prediction = qw_MF_krig_test_predict,
    fidelity = 'MF',
    model = 'Kriging',
    variable = 'Heat Transfer')

# [MF_NN_NRMSE_pressure, MF_NN_R2_pressure] = errorMetrics(
#     truth = p_MF_test_truth,
#     prediction = p_MF_NN_test_predict,
#     fidelity = 'MF',
#     model = 'NN')

# [MF_NN_NRMSE_heatTransfer, MF_NN_R2_heatTransfer] = errorMetrics(
#     truth = qw_MF_test_truth,
#     prediction = qw_MF_NN_test_predict,
#     fidelity = 'MF',
#     model = 'NN')

# %% [markdown]
# ## MF Neural Network

# %% [markdown]
# ### Assemble Data, Build Model

# %%
LowFidelityNN_MFinput = LF_NN.predict(X)
multiFidelityNN_input = np.concatenate((X,LowFidelityNN_MFinput), axis = 1)
multiFidelityNN_input.shape                                    

# %%
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

# %%
multiFidelityNN_outputFlat = np.hstack(multiFidelityNN_output)
multiFidelityNN_outputFlat.shape

# %%
##### SKLEARN DATA SPLIT 
random_state = 30 # used for reproducibility of data splits

X_mf = multiFidelityNN_input
y_mf = multiFidelityNN_outputFlat
Y_mf_names = np.hstack(outputVarNames)
X_names = np.hstack(inputTrainingNames)

X_mf_train, X_mf_test, y_mf_train, y_mf_test, M_inf_train, M_inf_test = train_test_split(
    X_mf, y_mf, M_inf, test_size=0.20, random_state=random_state)

X_mf_test, X_mf_val, y_mf_test, y_mf_val, M_inf_test, M_inf_val = train_test_split(
    X_mf_test, y_mf_test, M_inf_test, test_size=0.50, random_state=random_state)

# M_inf_train, M_inf_test = train_test_split(M_inf,test_size=0.20,random_state=random_state) # used in plotting
# M_inf_test, M_inf_val = train_test_split(M_inf_test,test_size=0.50,random_state=random_state) # used in plotting

print("Multi-fidelity X_train shape: {}".format(X_mf_train.shape))
print("Multi-fidelity X_test shape: {}".format(X_mf_test.shape))
print("Multi-fidelity X_val shape: {}".format(X_mf_val.shape))
print("Multi-fidelity y_lf_train shape: {}".format(y_mf_train.shape))
print("Multi-fidelity y_lf_test shape: {}".format(y_mf_test.shape))
print("Multi-fidelity y_lf_val shape: {}".format(y_mf_val.shape))
print(f"concatenation order: {X_names}")
print(f"concatenation order: {Y_mf_names}")

# %%
if MFNNTrain: 

    multiFidelityLayerSizeList = [1000, 500, 500] #input the number of neurons per layer. Length of this list indicates number of hidden layers
    learningRate = 1.0e-3 #From Deep Learning w/ Python (Chollet)
    regType = keras.regularizers.L2
    regValue = 1.0e-6
    hiddenLayerActivation = tf.nn.tanh
    outputLayerActivation = tf.nn.leaky_relu
    kernelInitializer = tf.keras.initializers.GlorotUniform()
    optimizer = tf.keras.optimizers.Adamax
    numEpochs = 10000
    myBatchSize = None
    validSplit = None
    loss = 'mse'
    validData = (X_mf_test, y_mf_test)

    callbacks_list = None

    MF_NN = None #sometimes remnants of previously trained models can hang around, it's best 
                #to clear the variable first 

    MF_NN = build_model_parameterized(
        input_data = X_mf_train, 
        output_data = y_mf_train,
        layerSizeList = multiFidelityLayerSizeList, 
        rate = learningRate, 
        regType = regType, 
        regValue = regValue,
        hiddenLayerActivation = hiddenLayerActivation,
        outputLayerActivation = outputLayerActivation,
        kernelInitializer = kernelInitializer,
        optimizer = optimizer,
        loss = loss)
try:
    tf.keras.utils.plot_model(MF_NN,show_shapes=True)
except:
    print('Cannot show model graph, tf.keras.utils.plot_model not loaded')
MF_NN.summary()

# %% [markdown]
# ### Train Model

# %%
#Initialize hyperparamters for Single Fidelity Model
if MFNNTrain: 

    MF_NN_epochs = None
    MF_NN_history = None
    print('MF NN training start')
    start = time.time()
    MF_NN_epochs, MF_NN_history = train_model_all_fidelity(
        model = MF_NN, 
        input_data = X_mf_train, 
        output_data = y_mf_train,
        numEpochs = numEpochs, 
        myBatchSize = myBatchSize,
        validData = validData,
        callbacks_list= callbacks_list)
    end = time.time()
    print('MF NN training complete. Time to train:')
    print(round((end-start),4))

# %% [markdown]
# ### Save Model

# %%
if MFNNSave:
    saveNN(fidelityLevel='MF')

# %%
# build this one so you just type the date and it populates all the strings/pkls/paths, etc. 

if MFNNLoad: 
    loadNN(
        neuralNetFolderName = 'MF_NN_2022-XX-XX'
        )

# %% [markdown]
# ### Generate Predicitions

# %%
mseNames = ["mean_squared_error",
            'val_mean_squared_error'
            ]
colorList = [ 'k', 'r']

plotTrainAndTestLoss(historyDict = MF_NN_history,
                     mseNames = mseNames,
                     epochRangeBegin = 0,
                     epochRangeEnd = numEpochs,
                     colorList = colorList,
                     fidelityLevel = 'Multi Fidelity')

# %%
generateInverseTransformedPredictions(
    X_train = X_mf_train,
    X_test = X_mf_test,
    y_train = y_mf_train,
    y_test = y_mf_test,
    method = 'NN',
    fidelityLevel = 'MF'
    )

# %% [markdown]
# ### Analyze Predictions (One to One Comparison)

# %%
oneToOnePlotTool(
    method='Neural Network', 
    desiredNumCasesForPlot=12, 
    X_test=X_test, 
    qw_prediction = qw_MF_NN_test_predict, 
    qw_truth = qw_MF_test_truth, 
    p_prediction = p_MF_NN_test_predict, 
    p_truth = p_MF_test_truth)

oneToOneVisualizationPlotAllData(
    case = 8, 
    qw_test_predict = qw_MF_NN_test_predict,
    p_test_predict = p_MF_NN_test_predict,
    qw_test_truth = qw_MF_test_truth, 
    p_test_truth = p_MF_test_truth, 
    M_inf_test = M_inf_test,
    method = 'Neural Network')

# %%
[MF_krig_NRMSE_pressure, MF_krig_R2_pressure] = errorMetrics(
    truth = p_MF_test_truth,
    prediction = p_MF_krig_test_predict,
    fidelity = 'MF',
    model = 'Kriging',
    variable = 'Pressure')

[MF_krig_NRMSE_heatTransfer, MF_krig_R2_heatTransfer] = errorMetrics(
    truth = qw_MF_test_truth,
    prediction = qw_MF_krig_test_predict,
    fidelity = 'MF',
    model = 'Kriging',
    variable = 'Heat Transfer')

[MF_NN_NRMSE_pressure, MF_NN_R2_pressure] = errorMetrics(
    truth = p_MF_test_truth,
    prediction = p_MF_NN_test_predict,
    fidelity = 'MF',
    model = 'NN',
    variable = 'Pressure')

[MF_NN_NRMSE_heatTransfer, MF_NN_R2_heatTransfer] = errorMetrics(
    truth = qw_MF_test_truth,
    prediction = qw_MF_NN_test_predict,
    fidelity = 'MF',
    model = 'NN',
    variable = 'Heat Transfer')


