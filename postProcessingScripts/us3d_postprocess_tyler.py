import code 
import pickle
import os
import numpy as np
import scipy
from scipy.interpolate import Rbf

path = "/home/tyty/Desktop/CFD_Files/AFRL_Scholars/1st_Round_Ls"
os.chdir(path)
retval = os.getcwd()
# print("Current working directory %s" % retval)
dirList = list(os.listdir(path))
len(dirList)
dirListRun = []
for entry in dirList: 
    if entry.startswith('Run'):
        dirListRun.append(entry)
dirList = None

for dirName in dirListRun:
    path = "/home/tyty/Desktop/CFD_Files/AFRL_Scholars/1st_Round_Ls"
    path += '/' + dirName
    os.chdir(path)
    fluidFileName='./wall.dat'

    # Lets work on first reading in the fluid data, parsing it correctly, and computing the cell
    # centers that the heat flux is being reported at

    header = open(fluidFileName, mode='r').readlines(50)
    fluidVar = header[0].strip().lstrip('VARIABLES=')
    fluidVarList = fluidVar.split(',')
    # Add variables for the cell centered data to be calculated 

    nextLine = header[1].strip().split(',')
    nN = int(nextLine[0].lstrip('ZONE N='))
    nE = int(nextLine[1].lstrip(' E='))
    # code.interact(local=locals())

    # Lets get the fluid data into a dictionary
    fluidVarData = np.genfromtxt(fluidFileName, skip_header=4, skip_footer=nE)
    fluidDict = {}

    # Get any node centered data first
    numNodeData = 3
    for i in range(numNodeData):
        x = i + 1
        fluidDict[fluidVarList[i]] = fluidVarData[i*nN:x*nN]

    # w get get the cell centered data
    # This is super jank sorry loool
    numVarData = int(header[1].split(',')[4].strip(']=CELLCENTERED)\n').strip('[4-'))
    count = 0
    for j in range(numNodeData, numVarData):
        start = nN*numNodeData + (count*nE)
        end   = start + nE
        fluidDict[fluidVarList[j]]  = fluidVarData[start:end]
        count += 1

    # Now the connectivity data
    numCellData = numVarData - numNodeData
    skip = (nN * numNodeData) + nE * numCellData + 4
    conn_list = np.genfromtxt(fluidFileName, skip_header=skip).astype(int)

    fluidDict['conn'] = conn_list
    # code.interact(local=locals())

    fluidVarList.extend(['x_cc', 'y_cc', 'z_cc'])
    xs    = fluidDict['x']
    ys    = fluidDict['y']
    zs    = fluidDict['z']

    x_cc = np.zeros(nE)
    y_cc = np.zeros(nE)
    z_cc = np.zeros(nE)

    # Perform the calculations to compute the cell centers of the CFD mesh
    for i in range(nE):
        nodes = fluidDict['conn'][i]

        for j in range(len(nodes)):
            x_cc[i] = x_cc[i] + xs[nodes[j]-1]
            y_cc[i] = y_cc[i] + ys[nodes[j]-1]
            z_cc[i] = z_cc[i] + zs[nodes[j]-1]
        x_cc[i] = x_cc[i]/4
        y_cc[i] = y_cc[i]/4
        z_cc[i] = z_cc[i]/4

    # Store this data to the CFD python dictionary
    fluidDict['x_cc'] = x_cc
    fluidDict['y_cc'] = y_cc
    fluidDict['z_cc'] = z_cc
    # code.interact(local=locals())

    #Now write out the us3d dictionary to a pickle
    f = open('./fluid_dict.pkl', 'wb')
    pickle.dump(fluidDict, f)
    f.close()