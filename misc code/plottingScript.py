# %%
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'# if you wish to supress TensorFlow debug info, set to 3
# import code
import glob
import pickle
import numpy as np
import warnings; warnings.simplefilter('ignore', np.RankWarning)
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300; plt.rcParams['axes.labelsize'] = 'xx-large';
from matplotlib.patches import Rectangle
import math

# %%

os.chdir('/mnt/c/Users/tyler/Desktop/mutli fidelity modeling/')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


with open('sizeAndSpeedCompDict_2022-09-06.pkl', "rb") as f:
    NNsizeandspeedcompdict = pickle.load(f)

namesList = 'NNmodelParamsList, NNtrainDataParamsList,NNtrainDataSizeOnDisk, NNmodelSizeOnDiskList, NNtrainTimeList'
namesList = namesList.replace(" ", "")
splitNamesList = namesList.split(',')

for i, name in enumerate(splitNamesList):
    globals()[name] = NNsizeandspeedcompdict['data'][i]

with open('krigsizeAndSpeedCompDict_2022-10-12.pkl', "rb") as f:
    krigSizeandspeedcompdict = pickle.load(f)

namesList = 'krigModelParamsList, krigTrainDataParamsList, krigTrainDataSizeOnDisk, krigModelSizeOnDiskList, krigTrainTimeList'
namesList = namesList.replace(" ", "")
splitNamesList = namesList.split(',')

for i, name in enumerate(splitNamesList):
    globals()[name] = krigSizeandspeedcompdict['data'][i]

linewidth = 3
ticksize = 23
sizeFactor = 1.

plt.rcParams["figure.figsize"] = (8,6)

fig, ax = plt.subplots(constrained_layout=True)

ax.plot(krigTrainDataParamsList, np.float_(krigModelSizeOnDiskList)*1e-6, label="Kriging",linewidth=linewidth)
ax.plot(NNtrainDataParamsList,np.float_(NNmodelSizeOnDiskList) *1e-6,label="Neural Network",linewidth=linewidth,color='firebrick')
ax.plot(NNtrainDataParamsList, np.float_(NNtrainDataSizeOnDisk)*1e-6, label="CFD Data", linewidth=linewidth,color='black')

ax.set_xlabel("Number of Training Parameters",fontsize=ticksize)
ax.set_ylabel("Size on Disk (MB)",fontsize=ticksize)
ax.legend(fontsize='xx-large')
ax.tick_params(axis='both', labelsize=ticksize)

def param2cases(x):
    return x/2568

def cases2params(x):
    return x*2568

secax = ax.secondary_xaxis('top', functions =(param2cases, cases2params))
secax.set_xlabel('Number of RANS Cases (Cone/Flare Wall Data)',fontsize=ticksize)
secax.tick_params(labelsize=ticksize)

def saveVectorizedFigures(name, print_pdf,print_svg):
    paperFiguresPath = os.path.normpath("/mnt/c/Users/tyler/Desktop/mutli fidelity modeling/figures/AIAA_aviation")
    os.chdir(paperFiguresPath)
    if print_pdf:
        pdfName = name+'.pdf'
        fig.savefig(pdfName, format='pdf')
    if print_svg:
        svgName = name+'.svg'
        fig.savefig(svgName, format='svg')

saveVectorizedFigures(name = 'OnlineStorageRequirement_v2',print_pdf=True,print_svg=True)
os.chdir('/mnt/c/Users/tyler/Desktop/mutli fidelity modeling/')
