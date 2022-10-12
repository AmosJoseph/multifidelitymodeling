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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# %%
def sigmoid(x):
    s=1/(1+np.exp(-x))
    # ds=s*(1-s)  
    return s

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    # dt=1-t**2
    return t

def relu(x):
    r = np.maximum(0,x)
    return r

x=np.arange(-3,3,0.01)

y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

plt.rcParams['lines.linewidth'] = 2.5
fig,ax = plt.subplots()
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.plot(x,y_sigmoid,label=r"Sigmoid $\left(\frac{1}{1+ e^{-x}}\right)$", color='firebrick')
ax.plot(x,y_tanh,label=r"Tanh ($\textrm{tanh}(x)$)",color = 'grey',linestyle=(0, (5, 1)))
ax.plot(x,y_relu,label=r"Relu ($\textrm{max}(0,x)$)",color='black',linestyle=(0, (3, 1, 1, 1))) 
ax.legend(bbox_to_anchor=(-0.05, 1), loc='upper left',fontsize='x-large')


def saveVectorizedFigures(name, print_pdf,print_svg):
    paperFiguresPath = os.path.normpath("/mnt/c/Users/tyler/Desktop/mutli fidelity modeling/figures/AIAA_aviation")
    os.chdir(paperFiguresPath)
    if print_pdf:
        pdfName = name+'.pdf'
        fig.savefig(pdfName, format='pdf')
    if print_svg:
        svgName = name+'.svg'
        fig.savefig(svgName, format='svg')

saveVectorizedFigures(name = 'activationFunctions',print_pdf=True,print_svg=True)
