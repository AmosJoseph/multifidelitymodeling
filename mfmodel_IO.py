def saveVersionedPickle(filename, objectToSave, path):
    baseName = filename
    counter = 2
    ext = '.pkl'
    dt = str(datetime.date.today())
    while os.path.exists('./' + filename + '_' + dt + ext):
        filename = baseName + '_v' + str(counter)
        counter += 1

    filename += '_' + dt + ext
    os.chdir(path)
    pickle.dump(objectToSave, open(filename, 'wb'))
    os.chdir(path)

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def variableChecker(stringToTest):
    if stringToTest in globals():
        print('Global variable')
    else: 
        print('Not global variable')

    if stringToTest in locals():
        print('Local variable')
    else: 
        print('Not local variable')

def saveNN(fidelityLevel):

    os.chdir(path)
    os.chdir(modelDir + '/' + NNDir)
    kerasFolderName = fidelityLevel + '_NN_'
    dt = str(datetime.date.today())
    kerasFolderName += dt

    kerasPath = makeNewUniqueDir(baseName = kerasFolderName)


    modelName = fidelityLevel + '_NN'
    model = globals()[modelName]
    tf.get_logger().setLevel('WARNING')
    model.save(kerasPath)
    
    os.chdir(kerasPath)
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

    os.chdir(path)

def loadNN(neuralNetFolderName):
    # Loading the NN is a bit easier-- but you'll need to specify the path. An example path is included already, 
    # which will need to be edited if you wish to load a different model. 
    os.chdir(path)
    loadFolderPath = path + '\\\\' + modelDir + '\\\\' + NNDir + '\\\\' + neuralNetFolderName
    loadFolderPath = os.path.join(path, modelDir, NNDir, neuralNetFolderName)
    loadFolderPath = os.path.normpath(loadFolderPath)
    os.chdir(loadFolderPath)
    loadedModelName = neuralNetFolderName[0:2] + '_NN'

    globals()[loadedModelName] = keras.models.load_model(loadFolderPath)

    ### Load History and Epochs

    desiredLoadedEpochsName = loadedModelName + '_epochs'
    desiredLoadedHistoryName = loadedModelName + '_history'

    epochsFileName = desiredLoadedEpochsName + neuralNetFolderName[5:] + '.pkl'
    historyFileName =  desiredLoadedHistoryName + neuralNetFolderName[5:] + '.pkl'

    globals()[desiredLoadedEpochsName] = pickle.load(open(epochsFileName, 'rb'))
    globals()[desiredLoadedHistoryName] = pickle.load(open(historyFileName, 'rb'))

    print(loadedModelName + ' loaded!!!!')
    os.chdir(path)

def loadConvStudyNN(convStudyFolderName, fidelityLevel, layerConfiguration):
 
    os.chdir(path)

    #generate layer config string for directory search 
    s = str(layerConfiguration)
    originalLayerConfigString = s
    disallowed_characters = '[] '

    for char in disallowed_characters:
        s = s.replace(char, '')
    s = s.replace(',','_')
    layerConfigString = s

    loadFolderPath = path + '\\\\' + modelDir + '\\\\' + NNDir + '\\\\' + convStudyDir + '\\\\' + convStudyFolderName
    os.chdir(loadFolderPath)
    loadedModelName = fidelityLevel + '_NN'
    loadedDictName = fidelityLevel + 'convStudyDict'
    pklName = glob.glob('*.pkl')
    if len(pklName) > 1:
        raise Exception('ERROR: More than one .pkl file in current directory. Remove extra .pkl or load all files manually')
    pklName = pklName[0]

    globals()[loadedDictName] = pickle.load(open(pklName, 'rb'))

    modelDirKeywords = layerConfigString + fidelityLevel + "*"
    modelDirName = glob.glob(modelDirKeywords)
    if len(modelDirName) > 1:
        raise Exception('ERROR: More than one directory with ' + str(layerConfiguration) + 'configuration. Remove extra directory or load manually')
    modelDirName = modelDirName[0]

    loadFolderPath = os.path.join(loadFolderPath, modelDirName)

    globals()[loadedModelName] = keras.models.load_model(loadFolderPath)

    ### Load History and Epochs

    loadedEpochsName = loadedModelName + '_epochs'
    loadedHistoryName = loadedModelName + '_history'

    globals()[loadedEpochsName] = globals()[loadedDictName][originalLayerConfigString]['epochs']
    globals()[loadedHistoryName] = globals()[loadedDictName][originalLayerConfigString]['history']

    print(loadedModelName + ' loaded!!!!')
    print(loadedEpochsName + ' loaded!!!!')
    print(loadedHistoryName + ' loaded!!!!')
    os.chdir(path)

def saveKrig(fidelityLevel):
    os.chdir(path)
    os.chdir(modelDir + '/' + krigDir)
    modelName = fidelityLevel + '_krig'
    model = globals()[modelName]
    filename = modelName + '_'
    dt = str(datetime.date.today())
    ext = '.sav'
    filename += dt + ext
    pickle.dump(model, open(filename, 'wb'))
    os.chdir(path)

def loadBatchTest(filename,DFname,printDF):
    os.chdir(path)
    os.chdir(modelDir + '/' + NNDir)
    globals()[DFname] = pd.read_pickle("./"+filename)  
    print('Loaded: ' ,DFname)
    if printDF:
        print(globals()[DFname])
    os.chdir(path)

def loadTunerPKL(filename):
    os.chdir(path)
    os.chdir(modelDir + '/' + NNDir)
    globals()['tuner'] = pd.read_pickle("./"+filename)  
    print('Loaded: ' ,filename)
    os.chdir(path)

def requiredMemoryCalculator(sizeData):
    requiredMem = sizeData*8 * 9.31e-10
    print('Memory required for Kriging: ', round(requiredMem,4), 'Gigabytes')

def loadKrigOpt(filename,dictName,printKeys):
    os.chdir(path)
    os.chdir(modelDir + '/' + krigDir)
    globals()[dictName] = pd.read_pickle("./"+filename)  
    print('Loaded: ' ,dictName)
    if printKeys:
        print(globals()[dictName].keys())
    os.chdir(path)

def saveTuner(tuner, fidelityLevel):
    os.chdir(path)
    os.chdir(modelDir + '/' + NNDir)
    filename = 'tunerPickle_' + fidelityLevel + '_'
    dt = str(datetime.date.today())
    ext = '.pkl'
    filename += dt + ext
    f = open(filename, 'wb')
    pickle.dump(tuner, f)
    f.close()
    print(filename, 'saved at location: ',path, modelDir ,'\\' ,NNDir)
    os.chdir(path)


def makeNewUniqueDir(baseName):
    counter = 2
    newDir = baseName

    while os.path.exists(newDir):

        newDir = baseName + '_v' + str(counter)
        counter += 1

    os.mkdir(newDir)
    return newDir

#I wrote this function like a moron. There is a nice way to make this recursive, take a look at the function measures the memory of a directory. This works for what I wanted it to do though. :-) 
def twoLevelDictStructurePrint(userDict):
    for key in userDict.keys():
        print(key)
        try:
            keyList = [item for item in userDict[key].keys()]
            print("    ", keyList)
            for item in keyList:
                nestedList = [item for item in userDict[key][item]]
                print("        ",nestedList)
        except:
            pass