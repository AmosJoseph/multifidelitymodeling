def percentdifferencecalc(x1,x2):
    mean = np.mean((x1,x2),axis=0)
    percentDifference = abs(((x1-x2)/mean)*100)
    return percentDifference

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
    hiddenLayerActivation, outputLayerActivation, outputLayerDataType, kernelInitializer,
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
        name = 'outputlayer',
        dtype=outputLayerDataType
        )(x)
    
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer=optimizer(learning_rate=rate),
             loss = loss,
             metrics = [tf.keras.metrics.MeanSquaredError(),
                       tf.keras.metrics.RootMeanSquaredError(),])
                       # "mae"])
    
    return model


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

def errorMetrics(truth,prediction,fidelity,model, variable, verbose):
    rmse = mean_squared_error(truth, prediction, squared=False)
    ybar = truth.max()-truth.min()
    nrmse = round(rmse/ybar,5)
    R2 = round(r2_score(truth, prediction),4)
    if verbose: 
        print(fidelity + ' ' + model + ' ' + variable + ' ' + 'R2: ' + str(R2) )
        print(fidelity + ' ' + model + ' ' + variable + ' ' + 'NRMSE: ' + str(nrmse * 100) + '%\n' )
    return nrmse, R2

def krigTrain(X_train,y_train, fidelityLevel, kernel, n_restarts,verbose):
    
    modelName = fidelityLevel + '_krig'
    
    globals()[modelName] = None
    globals()[modelName] = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=n_restarts)
    if verbose:
        print(modelName + ' training begin')
    start = time.time()
    globals()[modelName].fit(X_train, y_train)

    end = time.time()
    if verbose:
        print(modelName + ' training end')
        print(fidelityLevel + ' Kriging train time: %0.7f seconds' % (end-start) )
        print('Model name: ', modelName)
    print('Original kernel:' , kernel)
    print('Optimized kernel:' , globals()[modelName].kernel_ )
    print('_'*25)

def generateInverseTransformedPredictions(X_train,X_test,y_train,y_test,method,fidelityLevel,truncate,verbose):
    # Need to add validation data eventually
    if verbose:
        print('Method: ' + method + ', Fidelity Level: ' + fidelityLevel)
        print('_'*25)
    
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

    globals()[dataSplitNames[0][1:]] = globals()[modelName].predict(X_test)
    globals()[dataSplitNames[1][1:]] = globals()[modelName].predict(X_train)


    globals()[dataSplitNames[2][1:]] = y_test
    globals()[dataSplitNames[3][1:]] = y_train
    #######################################

    variableNameList =[] #to be used for the "un-truncating"

    for i in np.arange(len(outputVarNames)):
        for j in np.arange(numDataSplit):
            #Input Data Split
            globals()[outputVarNames[j] + dataSplitNames[i]] = np.hsplit(globals()[dataSplitNames[i][1:]],numDataSplit)[j]
            variableNameList.append(outputVarNames[j] + dataSplitNames[i])
            if verbose:
                print(dataSplitNames[i][1:] + ' part ' + str(j+1) + '--> ' + outputVarNames[j] + dataSplitNames[i])
            globals()[outputVarNames[j] + dataSplitNames[i+2]] = np.hsplit(globals()[dataSplitNames[i+2][1:]],numDataSplit)[j]
            variableNameList.append(outputVarNames[j] + dataSplitNames[i+2]) 
            if verbose:
                print(dataSplitNames[i+2][1:] + ' part ' + str(j+1) + '--> ' + outputVarNames[j] + dataSplitNames[i+2])
    
    if verbose:
        print(variableNameList)
        X_predictionSpeed = np.tile(X_train,(2,1))
        frequencyTempList = []
        predictionTimeTempList = []
        numTestCases = X_predictionSpeed.shape[0]
        for _ in np.arange(0,200):
            predictionStart = time.time()
            globals()[modelName].predict(X_predictionSpeed)
            predictionEnd = time.time()
            predictionTime = predictionEnd-predictionStart
            predictionTimePerCase = predictionTime/numTestCases
            predictionTimeTempList.append(predictionTimePerCase)
            frequency = 1/((predictionTime)/numTestCases)
            frequencyTempList.append(frequency)
        frequencyTestAverage = np.mean(frequencyTempList)
        predictionTimeAverage = np.mean(predictionTimeTempList)
        print('_'*15)
        print('Prediction frequency: %0.5f Hz. Prediction time per case: %0.7f seconds' \
            % (frequencyTestAverage, predictionTimeAverage) )
        print('_'*15)
        

    if truncate: 
        leftFluidScalarDistributionLength = globals()[outputVarNames[0]][0,:xSpotLeft].shape[0]
        rightFluidScalarDistributionLength =  globals()[outputVarNames[0]][0,xSpotLeft:].shape[0]
        if verbose:
            print("leftFluidScalarDistributionLength:", leftFluidScalarDistributionLength)
            print("rightFluidScalarDistributionLength:", rightFluidScalarDistributionLength)

        for var in variableNameList:
            temp_left = [np.tile(entry, leftFluidScalarDistributionLength) for entry in globals()[var][:,0] ]
            temp_right = [np.tile(entry,rightFluidScalarDistributionLength) for entry in globals()[var][:,1] ]
            temp_stacked = np.hstack((temp_left,temp_right))
            globals()[var] = temp_stacked
            if verbose:
                print(var, " shape: ", globals()[var].shape)
########### Inverse Scale ###########

    if fidelityLevel == 'LF':

        for name in LFoutputVarNames:
            ScalerName = name + '_OutputScaler'
            for tail in dataSplitNames:
                fullName = name[0:-3] + tail
                globals()[fullName] = globals()[ScalerName].inverse_transform(globals()[fullName])
                if verbose:
                    print(fullName + ' = ' + ScalerName + '.inverse_transform(' +  fullName + ')')
                
    else:
        
        for name in outputVarNames:
            ScalerName = name + '_OutputScaler'
            for tail in dataSplitNames:
                fullName = name + tail
                globals()[fullName] = globals()[ScalerName].inverse_transform(globals()[fullName])
                if verbose:
                    print( fullName + ' has been inverse transformed using ' + ScalerName + '! It is called ' + fullName)

def optimizeKrig(kernelList, X_train, y_train, X_test, y_test, fidelityLevel, method, n_restarts, verbose):
    startOpt = time.time()
    if method == 'kriging' or 'Kriging':
        methodShortName = 'krig'
    modelName = fidelityLevel + '_' + methodShortName
    errorDict = {'Original Kernel':[],'Optimized Kernel':[],'NRMSE (Pressure)':[], 'NRMSE (Heat Transfer)':[],'R^2 (Pressure)':[], 'R^2 (Heat Transfer)':[]}
    ############################### BOOKMARK
    modelDict = {}
    completionTime = []
    numIters = len(kernelList)

    dataSplitNames = [
    '_' + modelName + '_test_predict',
    '_' + modelName + '_train_predict',
    '_' + fidelityLevel + '_test_truth',
    '_' + fidelityLevel + '_train_truth'
    ]
    
    variableNameList = []
    for name in outputVarNames:
        for entry in dataSplitNames:
            variableNameList.append(name + entry)

    for i, kernel in enumerate(kernelList):
        print('_'*25)
        print('Optimization begin for kernel: ' + str(kernel))
        loopStart = time.time()
        currentIter = i+1
        krigTrain(
        X_train=X_train,
        y_train=y_train, 
        fidelityLevel= fidelityLevel,
        kernel=kernel, 
        n_restarts = n_restarts,
        verbose = False
        )

        generateInverseTransformedPredictions(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        method = methodShortName,
        fidelityLevel = fidelityLevel,
        verbose = False,
        truncate=downsampleLF
        )

        chosenVar = 'p'
        chosenSet = 'test'
        matches = [match for match in variableNameList if (chosenSet in match) and (chosenVar+'_' in match)]

        [NRMSE_pressure, R2_pressure] = errorMetrics(
        truth = globals()[matches[1]],
        prediction = globals()[matches[0]],
        fidelity = fidelityLevel,
        model = methodShortName,
        variable = 'Pressure',
        verbose = False)

        chosenVar = 'qw'
        chosenSet = 'test'
        matches = [match for match in variableNameList if (chosenSet in match) and (chosenVar+'_' in match)]

        [NRMSE_heatTransfer, R2_heatTransfer] = errorMetrics( 
        truth = globals()[matches[1]],
        prediction = globals()[matches[0]],
        fidelity = fidelityLevel,
        model = methodShortName,
        variable = 'Heat Transfer',
        verbose = False)
        
        errorDict['Original Kernel'].append(kernel)
        errorDict['Optimized Kernel'].append(globals()[modelName].kernel_)
        errorDict['NRMSE (Pressure)'].append(NRMSE_pressure)
        errorDict['NRMSE (Heat Transfer)'].append(NRMSE_heatTransfer)
        errorDict['R^2 (Pressure)'].append(R2_pressure)
        errorDict['R^2 (Heat Transfer)'].append(R2_heatTransfer)
        modelDict[str(kernel)] = (globals()[modelName].kernel_, globals()[modelName])
        print('Optimization complete for kernel: ' + str(kernel) + '-->' + str(globals()[modelName].kernel_) + '\n')
        print('Trained model stored in modelDict, key: ' + str(kernel))
        print('_'*25)

        loopEnd = time.time()
        currentIterTime = round((loopEnd-loopStart),4)
        completionTime.append(currentIterTime)
        estimatedRemainingTime = (numIters - currentIter)*np.mean(completionTime)
        if verbose:
            print('Iteration ', str(i+1), ' of ',str(numIters) ,' training complete, time elapsed: ', str(round(currentIterTime/60,4)), ' minutes ', )
            print('Estimated time to completion: ', str(round(estimatedRemainingTime/60,4)), ' minutes')

    os.chdir(path)
    os.chdir(modelDir + '/' + krigDir)

    errorCompDataFrame = pd.DataFrame.from_dict(errorDict)
    dt = str(datetime.date.today())
    dfName = fidelityLevel + '_errorCompDataFrame_' + dt + '.csv'
    errorCompDataFrame.to_csv(
        dfName,
        index = False
        )
        
    filename = 'optimizerPickle_' + fidelityLevel + '_'
    dt = str(datetime.date.today())
    ext = '.pkl'
    filename += dt + ext
    f = open(filename, 'wb')
    pickle.dump(errorCompDataFrame, f)
    f.close()
    print(filename, 'saved at location: ',path, modelDir ,'/' ,krigDir)
    os.chdir(path)
    endOpt = time.time()
    totalOptTime = round((endOpt - startOpt),4)/60
    print('Convergence study complete. Time elapsed: ' , str(totalOptTime), ' minutes.')

# throw in a little search algorithm that picks the best one and retrains based on that kernel. Maybe keeps the trained models in a dict, picks the good one, deletes the rest. 
    return errorCompDataFrame, errorDict, modelDict

def operatingConditions(case):
    gamma = 1.4 # perfect gas
    R_specific = 287.058
    T_inf = inputTemperature[case]
    u_inf = inputVelocity[case]
    a_inf = np.sqrt(gamma*R_specific*T_inf)
    M_inf = u_inf/a_inf

    print('Case number: ' + str(case))

    print(
        'Wall Temp: ' + str(round(inputWallTemp[case].item(), 2)) + ' K\n' + \
            'Freestream Temp: ' + str(round(inputTemperature[case].item(), 2)) + ' K\n' + \
                'Freestream Density: ' + str(round(inputDensity[case].item(), 2)) + ' kg/m3\n' + \
                    'Freestream Velocity: ' + str(round(inputVelocity[case].item(), 2)) + ' m/s\n' + \
                        'Mach Number: ' + str(M_inf)
        )

def batchTest(model,fidelityLevel, batchSizeMultiples, input_data, output_data,validData,numEpochs):
    batchSizeList = []
    for multiple in np.arange(start=1,stop = batchSizeMultiples):
        batchSizeList.append(32*multiple)

    batchNumberList = []
    timeList = []
    timeDict = dict()

    for batch in batchSizeList:
        # LF_NN_epochs = None
        # LF_NN_history = None
        print('Batch size: ',str(batch), ' training start')
        start = time.time()
        train_model_all_fidelity(
            model = model, 
            input_data = input_data, 
            output_data = output_data,
            numEpochs = numEpochs, 
            myBatchSize = batch,
            validData = validData,
            callbacks_list= callbacks_list)
        end = time.time()
        batchNumberList.append(batch)
        timeList.append(round((end-start),4))
        print('Batch size: ',str(batch), ' training end. Time elapsed: ',str(round((end-start),4)) )

    timeDict["Batch"] = batchNumberList
    timeDict["Time"] = timeList
    timeDF = pd.DataFrame(data=timeDict)

    os.chdir(path)
    os.chdir(modelDir + '/' + NNDir)
    filename = 'timePickle_' + fidelityLevel + '_'
    dt = str(datetime.date.today())
    ext = '.pkl'
    filename += dt + ext
    f = open(filename, 'wb')
    pickle.dump(timeDF, f)
    f.close()
    print(filename, 'saved at location: ',path, modelDir ,'/' ,NNDir)
    os.chdir(path)

def neuralNetworkConvergence(
    fidelityLevel,convStudyLayerList, hyperparamDict,X_train,y_train, validData, callbacks_list, verbose, showConvPlot, saveConvPlot, showMSEplot, saveMSEplot, showSpeedPlot, saveSpeedPlot
    ):

    convStudyTimeStart = time.time()
    numParamsList = []
    NNminMSElist = []
    completionTime = []
    predictionTimeList = []
    predictionTimePerCaseList = []
    frequencyList = []
    # fileSizeList = []
    dictName = fidelityLevel + 'convStudyDict'
    globals()[dictName] = dict()

    annotations = [str(item) for item in convStudyLayerList]
    depthList = [len(item) for item in convStudyLayerList]
    disallowed_characters = '[] '
    dirAnnotations = []
    for s in annotations:
        for char in disallowed_characters:
            s = s.replace(char, '')
        s = s.replace(',','_')
        dirAnnotations.append(s)

    numEpochs = hyperparamDict["numEpochs"]
    if quickTestRun:
        numEpochs = 3
    validData = validData
    X_predictionSpeed = np.tile(X_train,(2,1))
    callbacks_list = callbacks_list

    NN = None #sometimes remnants of previously trained models can hang around, it's best 
                #to clear the variable first 
    numIters = len(convStudyLayerList)
    currentConvStudyTopDir = fidelityLevel + time.strftime("%Y%m%d-%H%M%S")
    convStudyPath = os.path.join(path,modelDir,NNDir,convStudyDir)
    os.chdir(convStudyPath)
    os.mkdir(currentConvStudyTopDir)
    os.chdir(currentConvStudyTopDir)

    for i, layerSizeList in enumerate(convStudyLayerList):
        currentIter = i+1
        NN = build_model_parameterized(
            input_data = X_train, 
            output_data = y_train,
            layerSizeList = layerSizeList, 
            rate = hyperparamDict["learningRate"], 
            regType = hyperparamDict["regType"], 
            regValue = hyperparamDict["regValue"],
            hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"],
            outputLayerActivation = hyperparamDict["outputLayerActivation"],
            outputLayerDataType= 'float32',
            kernelInitializer = hyperparamDict["kernelInitializer"],
            optimizer = hyperparamDict["optimizer"],
            loss = hyperparamDict["loss"])

        NN_epochs = None
        NN_history = None
        print('Iteration ', str(currentIter), ' training start')
        start = time.time()
        NN_epochs, NN_history = train_model_all_fidelity(
            model = NN, 
            input_data = X_train, 
            output_data = y_train,
            numEpochs = numEpochs, 
            myBatchSize = hyperparamDict["myBatchSize"],
            validData = validData,
            callbacks_list= callbacks_list)

        minMSE = np.min(NN_history["val_mean_squared_error"])
        numParams = NN.count_params()

        frequencyTempList = []
        predictionTimeTempList = []
        numTestCases = X_predictionSpeed.shape[0]
        for _ in np.arange(0,200):
            predictionStart = time.time()
            NN.predict(X_predictionSpeed)
            predictionEnd = time.time()
            predictionTime = predictionEnd-predictionStart
            predictionTimeTempList.append(predictionTime)
            frequency = 1/((predictionTime)/numTestCases)
            frequencyTempList.append(frequency)
        frequencyTestAverage = np.mean(frequencyTempList)
        predictionTimeAverage = np.mean(predictionTimeTempList)


        predictionTimeList.append(predictionTimeAverage)
        predictionTimePerCase = (predictionTimeAverage)/numTestCases
        predictionTimePerCaseList.append(predictionTimePerCase)

        frequencyList.append(frequencyTestAverage)

        globals()[dictName][annotations[i]] = {
        'numParams' : numParams,
        'minMSE' : minMSE,
        'epochs' : NN_epochs,
        'history' : NN_history,
        }

        numParamsList.append(numParams)
        NNminMSElist.append(minMSE)
        # fileSizeList = []

        currentIterKerasDirName = dirAnnotations[i] + fidelityLevel + '_convStudyModel'
        # model is saved separately/outside of the dictionary, because the keras model objects doesn't play nice with pickle function. Throws an error about saving local functions. 

        kerasPath = os.path.join(convStudyPath,currentConvStudyTopDir, currentIterKerasDirName)
        tf.get_logger().setLevel('WARNING')
        NN.save(kerasPath)
        os.chdir(convStudyPath)

        end = time.time()
        currentIterTime = round((end-start),4)
        completionTime.append(currentIterTime)
        estimatedRemainingTime = (numIters - currentIter)*np.mean(completionTime)
        if verbose:
            print('Iteration ', str(i+1), ' of ',str(numIters) ,' training complete. Layer configuration: ', str(layerSizeList) ,' time elapsed: ', str(round(currentIterTime/60,4)), ' minutes ', )
            print('Estimated time to completion: ', str(round(estimatedRemainingTime/60,4)), ' minutes')
            print('Number of epochs: ', len(NN_epochs), 'original number of epochs: ', str(numEpochs))
    
    globals()[dictName]['numParamsList'] = numParamsList
    globals()[dictName]['NNminMSElist'] = NNminMSElist
    globals()[dictName]['predictionTimeList'] = predictionTimeList
    globals()[dictName]['predictionTimePerCaseList'] = predictionTimePerCaseList
    globals()[dictName]['frequencyList'] = frequencyList

    saveVersionedPickle(
    filename=dictName, 
    objectToSave=globals()[dictName],
    path = os.path.join(convStudyPath,currentConvStudyTopDir)
    )
    print('Dictionary name: ', dictName)
    os.chdir(path)
    convStudyTimeEnd = time.time()
    totalConvStudyTime = round((convStudyTimeEnd - convStudyTimeStart),4)/60

    print('Convergence study complete. Convergene study directory name: ',currentConvStudyTopDir, '. Time elapsed: ' , str(totalConvStudyTime), ' minutes. Beginning Plotting. ')

    if showConvPlot:
        plotNNConvergence(
            fidelityLevel = fidelityLevel,
            numParamsList = numParamsList,
            MSElist = NNminMSElist,
            convStudyLayerList = convStudyLayerList,
            savePlot = saveConvPlot
        )
    if showMSEplot: 
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle(fidelityLevel + " NN Convergence MSE", fontsize=18, y=0.95)
        mseNames = ["mean_squared_error",
                    'val_mean_squared_error'
                    ]
        colorList = [ 'k', 'r']
        epochRangeBegin = 0
        for i, annotation in enumerate(annotations):
            historyDict = globals()[dictName][annotation]['history']
            ax = plt.subplot(int(math.ceil(len(annotations)/2)),2, i+1)
            ax.set_title('Neural Network Hidden Layer Structure: ' + annotation)
            for color, mse in enumerate(mseNames):
                ax.semilogy(
                    range(1,len(historyDict[mse][epochRangeBegin:]) + 1),
                    historyDict[mse][epochRangeBegin:],
                    label=mse,linestyle="-", color=colorList[color]
                    )
        if saveMSEplot: 
            figName = 'NNMeanSquaredErrorConvStudy' + fidelityLevel + '_'
            dt = str(datetime.date.today())
            figName += dt
            os.chdir(path)
            os.chdir(figureDir)
            baseName = figName
            counter = 2
            while os.path.exists('./' + figName + '.png'):
                figName = baseName + '_v' + str(counter)
                counter += 1
            plt.savefig(figName)
            os.chdir(path)

    if showSpeedPlot:
        plt.rcParams["figure.figsize"] = (6.4,4.8) 
        # plt.figure()
        # plt.scatter(numParamsList, fileSizeList)
        # x_values = [np.min(numParamsList), np.max(numParamsList)]
        # y_values = [np.min(fileSizeList),np.max(fileSizeList)]
        # for i, label in enumerate(annotations):
        #     plt.annotate(label, (numParamsList[i], predictionTimeList[i]))
        # plt.plot(x_values,y_values,'bo', linestyle = '--')
        # plt.xlabel('Num of NN Params')
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.ylabel('File Size (MB)')
        # plt.show()
        plt.figure()
        numColors = (np.max(depthList) - np.min(depthList))+1
        colors = plt.cm.jet(np.linspace(0,1,numColors))

        for j, depth in enumerate(set(depthList)):
            depthIdx = [int(i) for i in range(len(depthList)) if depthList[i] == j+1]
            label = 'NN Depth: ' + str(depth)

            x = [numParamsList[i] for i in depthIdx]
            y = [frequencyList[i] for i in depthIdx]
            plt.scatter(x, y, label=label, color=colors[j])
            plt.xlabel('Num of NN Params')
            plt.ylabel('Prediction Frequency (Hz)')
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend()
        plt.show()

        if saveSpeedPlot: 
            figName = 'NNPredictionSpeed' + fidelityLevel + '_'
            dt = str(datetime.date.today())
            figName += dt
            os.chdir(path)
            os.chdir(figureDir)
            baseName = figName
            counter = 2
            while os.path.exists('./' + figName + '.png'):
                figName = baseName + '_v' + str(counter)
                counter += 1
            plt.savefig(figName)
            os.chdir(path)

    plt.rcParams["figure.figsize"] = (6.4,4.8) #reset to defaults 
    

def neuralNetworkSizeAndSpeedTest(numTestCases, numInputVars, numOutputVars, layerSizeList, hyperparamDict, showPlot):
    
    X_dummyData = np.random.rand(numTestCases, numInputVars)
    y_dummyData = np.random.rand(numTestCases, numOutputVars)

    numParamsList = []
    fileSizeList = []
    predictionTimeList = []
    predictionTimePerCaseList = []
    frequencyList = []
    dictName = 'SizeandSpeedStudyDict' 
    globals()[dictName] = dict()
    globals()[dictName]['layerSizeList'] = layerSizeList

    annotations = [str(item) for item in layerSizeList]
    print(annotations)

    learningRate = hyperparamDict["learningRate"]
    regType = hyperparamDict["regType"]
    regValue = hyperparamDict["regValue"]
    hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"]
    outputLayerActivation = hyperparamDict["outputLayerActivation"]
    kernelInitializer = hyperparamDict["kernelInitializer"]
    optimizer = hyperparamDict["optimizer"]
    numEpochs = 1000
    myBatchSize = hyperparamDict["myBatchSize"]
    loss = hyperparamDict["loss"]

    currentTestTopDir = 'sizeAndSpeedTest_' + time.strftime("%Y%m%d-%H%M%S")
    sizeAndSpeedPath = os.path.join(path,modelDir,NNDir,sizeAndSpeedDir)
    os.chdir(sizeAndSpeedPath)
    os.mkdir(currentTestTopDir)
    os.chdir(currentTestTopDir)
    currentTestTopPath = os.path.join(sizeAndSpeedPath, currentTestTopDir)

    for i, layerSizeList in enumerate(layerSizeList):
        currentIter = i+1
        NN = None #sometimes remnants of previously trained models can hang around, it's best to clear the variable first 
        NN = build_model_parameterized(
            input_data = X_dummyData, 
            output_data = y_dummyData,
            layerSizeList = layerSizeList, 
            rate = learningRate, 
            regType = regType, 
            regValue = regValue,
            hiddenLayerActivation = hiddenLayerActivation,
            outputLayerActivation = outputLayerActivation,
            kernelInitializer = kernelInitializer,
            optimizer = optimizer,
            outputLayerDataType= 'float32',
            loss = loss)

        print('Iteration ', str(currentIter), ' training start')

        train_model_all_fidelity(
            model = NN, 
            input_data = X_dummyData, 
            output_data = y_dummyData,
            numEpochs = numEpochs, 
            myBatchSize = myBatchSize,
            validData = validData,
            callbacks_list= None)

        numParams = NN.count_params()
        numParamsList.append(numParams)

        kerasPath = os.path.join(sizeAndSpeedPath,currentTestTopDir)
        os.chdir(kerasPath)
        os.mkdir('temp')
        os.chdir('temp')
        NN.save(os.getcwd())
        tempDirSize = (get_dir_size('.'))/(2**20)
        fileSizeList.append(tempDirSize)

        os.chdir(kerasPath)
        shutil.rmtree('temp')
        os.chdir(currentTestTopPath)
        
        predictionStart = time.time()
        NN.predict(X_dummyData)
        predictionEnd = time.time()
            
        predictionTime = predictionEnd-predictionStart
        predictionTimeList.append(predictionTime)

        predictionTimePerCase = (predictionTime)/numTestCases
        predictionTimePerCaseList.append(predictionTimePerCase)

        frequency = 1/((predictionTime)/numTestCases)
        frequencyList.append(frequency)
    
    globals()[dictName]['numParamsList'] = numParamsList
    globals()[dictName]['fileSizeList'] = fileSizeList
    globals()[dictName]['predictionTimeList'] = predictionTimeList
    globals()[dictName]['predictionTimePerCaseList'] = predictionTimePerCaseList
    globals()[dictName]['frequencyList'] = frequencyList

    currentTestTopPath = os.path.join(sizeAndSpeedPath,currentTestTopDir)
    
    saveVersionedPickle(
    filename=dictName, 
    objectToSave=globals()[dictName],
    path = currentTestTopPath
    )
    print('Dictionary name: ', dictName)
    

    if showPlot:
        plt.rcParams["figure.figsize"] = (6.4,4.8) 
        plt.scatter(numParamsList, fileSizeList)
        x_values = [np.min(numParamsList), np.max(numParamsList)]
        y_values = [np.min(fileSizeList),np.max(fileSizeList)]
        for i, label in enumerate(annotations):
            plt.annotate(label, (numParamsList[i], predictionTimeList[i]))
        plt.plot(x_values,y_values,'bo', linestyle = '--')
        plt.xlabel('Num of NN Params')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ylabel('File Size (MB)')
        plt.show()

        numColors = (np.max(depthList) - np.min(depthList))+1
        colors = plt.cm.jet(np.linspace(0,1,numColors))

        for j, depth in enumerate(set(depthList)):
            depthIdx = [int(i) for i in range(len(depthList)) if depthList[i] == j+1]
            label = 'NN Depth: ' + str(depth)

            x = [numParamsList[i] for i in depthIdx]
            y = [frequencyList[i] for i in depthIdx]
            plt.scatter(x, y, label=label, color=colors[j])
            plt.xlabel('Num of NN Params')
            plt.ylabel('Prediction Frequency (Hz)')
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend()
        plt.show()


    os.chdir(path)
    print('Convergence study complete. ')
    return globals()[dictName]

def highFidelityDataGenAndProcess(verbose=False,split=False):
    
    global inputTrainingData, inputTrainingNames,outputTrainingData, outputTrainingNames,M_inf, X, y , Y_names, X_names, originalIdx, X_train, X_test, y_train, y_test, M_inf_train, M_inf_test, trainIdx, testIdx, X_val, y_val, M_inf_val,  valIdx

    inputTrainingData = []
    inputTrainingNames = []

    for i, name in enumerate(inputVarNames):
        ScalerName = name + '_InputScaler'
        ScaledName = name + '_Scaled'
        InputDataName = 'input' + name
        globals()[ScalerName] = None
        globals()[ScalerName] = preprocessing.StandardScaler()
        globals()[ScaledName] = globals()[ScalerName].fit_transform(globals()[InputDataName])
        inputTrainingData.append(globals()[ScaledName])
        inputTrainingNames.append(ScaledName)
        max_element = str(round(np.max(globals()[ScaledName]),2))
        min_element = str(round(np.min(globals()[ScaledName]),2))
        if verbose: 
            print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

    outputTrainingData = []
    outputTrainingNames = []

    for i, name in enumerate(outputVarNames):
        ScalerName = name + '_OutputScaler'
        ScaledName = name + '_Scaled'
        OutputDataName = name
        globals()[ScalerName] = None
        globals()[ScalerName] = preprocessing.StandardScaler()
        globals()[ScaledName] = globals()[ScalerName].fit_transform(globals()[OutputDataName])
        outputTrainingData.append(globals()[ScaledName])
        outputTrainingNames.append(ScaledName)
        max_element = str(round(np.max(globals()[ScaledName]),2))
        min_element = str(round(np.min(globals()[ScaledName]),2))
        if verbose:
            print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

    gamma = 1.4 # perfect gas
    R_specific = 287.058

    T_inf = inputTemperature
    rho_inf = inputDensity
    u_inf = inputVelocity
    a_inf = np.sqrt(gamma*R_specific*T_inf)
    M_inf = u_inf/a_inf

    ##### SKLEARN DATA SPLIT 

    X = np.hstack(inputTrainingData)
    y = np.hstack(outputTrainingData)
    Y_names = np.hstack(outputTrainingNames)
    X_names = np.hstack(inputTrainingNames)
    originalIdx = np.arange(0,X.shape[0])
    if split:
        X_train, X_test, y_train, y_test, M_inf_train, M_inf_test, trainIdx, testIdx = train_test_split(
            X, y, M_inf, originalIdx, test_size=0.20, random_state=random_state)

        X_test, X_val, y_test, y_val, M_inf_test, M_inf_val, testIdx, valIdx = train_test_split(
            X_test, y_test, M_inf_test, testIdx, test_size=0.50, random_state=random_state)
        # M_inf_train, M_inf_test = train_test_split(M_inf,test_size=0.20,random_state=random_state) # used in plotting
        # M_inf_test, M_inf_val = train_test_split(M_inf_test,test_size=0.50,random_state=random_state) # used in plotting
        if verbose: 
            print('Input Data (stored in list inputTrainingData):\n')
            print('\nOutput Data (stored in list outputTrainingData):\n')
            print(str(np.shape(inputTrainingData)))
            print(str(np.shape(outputTrainingData)))
            print(inputTrainingNames)
            print(outputTrainingNames)
            print("X_train shape: {}".format(X_train.shape))
            print("X_test shape: {}".format(X_test.shape))
            print("y_train shape: {}".format(y_train.shape))
            print("y_test shape: {}".format(y_test.shape))
            print("X_val shape: {}".format(X_val.shape))
            print("y_val shape: {}".format(y_val.shape))
            print(f"concatenation order: {X_names}")
            print(f"concatenation order: {Y_names}")

def lowFidelityDataGenAndProcess(downsampleLF:bool, verbose=False):
    ## Input Conditions, low fidelity data generation

    # These probably should be renamed to be consistent with the input variables already created. 


    # inputVar list contains:  ['WallTemp', 'Density', 'Temperature', 'Velocity']

    gamma = 1.4 # perfect gas
    R_specific = 287.058
    cp1 = 1.005 #KJ/Kg*K, air at 251K

    LFinputVarNames = [
        'T_w',
        'rho_inf',
        'T_inf',
        'u_inf'
    ]

    LFbegin = time.time()
    for i, inputVarName in enumerate(LFinputVarNames): 
        globals()[inputVarName] = lowFidelityInputPoints[:,i].reshape(-1,1)
        if verbose:
            print(inputVarName, 'created , shape ', str(globals()[inputVarName].shape))
            print("Lower bound: ", globals()[inputVarName].min(), ". Upper bound: ", globals()[inputVarName].max())

    numCases = lowFidelityInputPoints.shape[0]
    if verbose:
        print('Num low-fidelity cases: ', str(numCases))

    a_inf = np.sqrt(gamma*R_specific*T_inf)
    M_inf = u_inf/a_inf
    P_inf = rho_inf*R_specific*T_inf
    mu_inf = mu_suth(T_inf)

    theta  = np.full((numCases,1),np.deg2rad(7))

    inputDataObliqueShock = M_inf,u_inf,T_inf,P_inf,rho_inf,a_inf,theta

    [*temp] = map(perfgas_oblique, M_inf,u_inf,T_inf,P_inf,rho_inf,a_inf,theta)
    obliqueShockResults = np.array(temp)

    # print(M_inf.shape,u_inf.shape,T_inf.shape,P_inf.shape,rho_inf.shape,a_inf.shape, theta.shape )
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
        globals()[name] = obliqueShockResults[:,i]
        # print(globals()[name].shape)
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

    # PressureForCPActual = p[:,xSpotNoMean].reshape(-1,1)
    # cp_actual = (PressureForCPActual - P_inf)/ (0.5*rho_inf*(u_inf**2))
    # cp_actual[389] = None # takes care of that one bad point

    #Shock Expansion For 40deg Section

    T_inf2 = T2
    # T_w = T_w
    rho_inf2 = rho2
    u_inf2 = M2*a2
    a_inf2 = a2
    M_inf2 = M2
    P_inf2 = P2
    mu_inf2 = mu_suth(T2)
    theta2  = np.full((numCases,1),np.deg2rad(33))

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
        globals()[name] = obliqueShockResults[:,i]
    xPressureWindowLeft = 2.353056 # elbow location
    xPressureWindowRight = 2.5039961 # end of cone

    xPressureWindowMid = 2.4
    xSpotLeft = (np.abs(x_cc_windowed[0,:] - xPressureWindowLeft)).argmin()
    xSpotRight = (np.abs(x_cc_windowed[0,:] - xPressureWindowRight)).argmin()

    # meanPressure40DegConeSection = np.median(p[:,xSpotLeft:xSpotRight], axis = 1).reshape(-1,1)
    # cp_actual2 = (meanPressure40DegConeSection - P_inf2)/ (0.5*rho_inf2*(u_inf2**2))
    # cp_actual2[389] = None # takes care of that one bad case

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

    # normalizedHFPressure = p/P03
    # normalizedLFPressure = p_lowFidelity_SE/P03
    # normalizedLFPressureNewt = p_lowFidelity_Newt/P03

    p_LF = p_lowFidelity_SE
    if verbose:
        print("p_LF shape: " ,p_LF.shape)

    ## ---- Eckert's Reference Temperature, Cone Example ----

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
    if verbose: 
        print("qw_LF shape: " ,qw_LF.shape)
    LFend = time.time()

    if verbose: 
        LFtotalTime = LFend - LFbegin
        print(f"time to generate: {LFtotalTime} seconds")
    LFoutputVarNames = ['qw_LF','p_LF']    

    totalParamsTrainData = 0
    for var in LFoutputVarNames:
        globals()[var] = locals()[var]
        totalParamsTrainData += globals()[var].shape[0] * globals()[var].shape[1]
        if verbose: 
            print(var, "globals:" , globals()[var].shape,"locals: ", locals()[var].shape)
    if verbose:
        print('Total number of parameters in training data: ', totalParamsTrainData)
        print("outlierRemoval criteria:", outlierRemoval, " and ", (locals()[LFoutputVarNames[0]].shape[0] == numCases))
        print(globals()[LFoutputVarNames[0]].shape[0], "== ", numCases)

    if outlierRemoval and (globals()[LFoutputVarNames[0]].shape[0] == numCases):
        if verbose:
            print('Entered outlier removal loop')
        casesToRemove = np.argwhere(np.isnan(qw_LF).any(axis=1))
        if numpPointsMultiplier == 2:
            casesToRemove = [648]
        
        for i, outputVarName in enumerate(LFoutputVarNames): 
            globals()[outputVarName] = np.delete(locals()[outputVarName], casesToRemove, axis=0)
            if verbose:
                print(outputVarName, 'new shape ', str(globals()[outputVarName].shape))

        for i, inputVarName in enumerate(LFinputVarNames): 
            globals()[inputVarName] = np.delete(globals()[inputVarName], casesToRemove, axis=0)
            if verbose:
                print(inputVarName, 'new shape ', str(globals()[inputVarName].shape))

        M_inf = np.delete(M_inf, casesToRemove, axis=0)
        if verbose:
            print('M_inf new shape ', str(M_inf.shape))
            print('Removed ', len(casesToRemove), ' case(s).')


    LFinputTrainingData = []
    LFinputTrainingNames = []

    if verbose:
        print('Input Data (stored in list inputTrainingData):\n')
    for i, name in enumerate(LFinputVarNames):
        ScalerName = name + '_InputScaler'
        ScaledName = name + '_Scaled'
        globals()[ScalerName] = None
        globals()[ScalerName] = preprocessing.StandardScaler()
        globals()[ScaledName] = globals()[ScalerName].fit_transform(globals()[name])
        LFinputTrainingData.append(globals()[ScaledName])
        LFinputTrainingNames.append(ScaledName)
        max_element = str(round(np.max(globals()[ScaledName]),2))
        min_element = str(round(np.min(globals()[ScaledName]),2))
        if verbose:
            print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)

    LFoutputTrainingData = []
    LFoutputTrainingNames = []

    if verbose:
        print('\nOutput Data (stored in list LFoutputTrainingData):\n')
    for i, name in enumerate(LFoutputVarNames):
        ScalerName = name + '_OutputScaler'
        ScaledName = name + '_Scaled'
        OutputDataName = name
        globals()[ScalerName] = None
        globals()[ScalerName] = preprocessing.StandardScaler()
        globals()[ScaledName] = globals()[ScalerName].fit_transform(globals()[OutputDataName])
        LFoutputTrainingData.append(globals()[ScaledName])
        LFoutputTrainingNames.append(ScaledName)
        max_element = str(round(np.max(globals()[ScaledName]),2))
        min_element = str(round(np.min(globals()[ScaledName]),2))
        if verbose:
            print(name + ' has been scaled! It is called ' + ScaledName + '. Min:' + min_element + '. Max:' + max_element)
    if verbose and not downsampleLF: 
            print("Number of input variables: ",  str(np.shape(LFinputTrainingData)[0] ) )
            print("Shape of LFinputTrainingData: ",  str(np.shape(LFinputTrainingData) ) )
            print("Number of output variables: ",  str(np.shape(LFoutputTrainingData)[0] ) )
            print("Number of variables in output distribution: ",  str(np.shape(LFoutputTrainingData)[2] ))
            print("Shape of LFoutputTrainingData: ",  str(np.shape(LFoutputTrainingData) ) )
            print("LFinputTrainingNames: ", LFinputTrainingNames)
            print("LFoutputVarNames: ", LFoutputVarNames)
            print("LFoutputTrainingNames: ", LFoutputTrainingNames)

    if downsampleLF: 
        ##################################################
        ####### Heat Flux and Pressure Downsample ########
        ################################################## 
        conePoint = 2.0
        flarePoint = 2.49
        xLocationPressureValue1 = (np.abs(x_cc_windowed[0,:] - conePoint)).argmin()
        xLocationPressureValue2 = (np.abs(x_cc_windowed[0,:] - flarePoint)).argmin()

        indices = [xLocationPressureValue1, xLocationPressureValue2]
        LFoutputTrainingData = []
        for name in LFoutputTrainingNames:
            globals()[name] = np.take(globals()[name], indices, axis=1)
            LFoutputTrainingData.append(globals()[name])
            if verbose:
                print(name, ' truncated. Added to LFoutputTrainingData')
        x_downsampledPressure = np.take(x_cc_windowed, indices, axis=1)
        if verbose:
            print("Number of input variables: ",  str(np.shape(LFinputTrainingData)[0] ) )
            print("Shape of LFinputTrainingData: ",  str(np.shape(LFinputTrainingData) ) )
            print("Number of output variables: ",  str(np.shape(LFoutputTrainingData)[0] ) )
            print("Number of variables in output distribution: ",  str(np.shape(LFoutputTrainingData)[2] ))
            print("Shape of LFoutputTrainingData: ",  str(np.shape(LFoutputTrainingData) ) )
            print("LFinputTrainingNames: ", LFinputTrainingNames)
            print("LFoutputVarNames: ", LFoutputVarNames)
            print("LFoutputTrainingNames: ", LFoutputTrainingNames)

    ##### SKLEARN DATA SPLIT 
    global X, y_lf, Y_lf_names,X_names,originalIdx,X_train, y_lf_train, M_inf_train, trainIdx, X_test, X_val, y_lf_test, y_lf_val, M_inf_test, M_inf_val, testIdx, valIdx


    X = np.hstack(LFinputTrainingData)
    y_lf = np.hstack(LFoutputTrainingData)
    Y_lf_names = np.hstack(LFoutputTrainingNames)
    X_names = np.hstack(LFinputTrainingNames)
    originalIdx = np.arange(0,X.shape[0])
    if verbose: 
        print("X.shape: ", X.shape)
        print("y_lf.shape: ", y_lf.shape)
        print("M_inf.shape: ", M_inf.shape)
        print("originalIdx.shape: ", originalIdx.shape)
    


    X_train, X_test, y_lf_train, y_lf_test, M_inf_train, M_inf_test, trainIdx, testIdx = train_test_split(
        X, y_lf, M_inf, originalIdx, test_size=0.20, random_state=random_state)

    X_test, X_val, y_lf_test, y_lf_val, M_inf_test, M_inf_val, testIdx, valIdx = train_test_split(
        X_test, y_lf_test, M_inf_test, testIdx, test_size=0.50, random_state=random_state)

    if verbose:
        print("Low fidelity X_train shape: {}".format(X_train.shape))
        print("Low fidelity X_test shape: {}".format(X_test.shape))
        print("Low fidelity X_val shape: {}".format(X_val.shape))
        print("Low fidelity y_lf_train shape: {}".format(y_lf_train.shape))
        print("Low fidelity y_lf_test shape: {}".format(y_lf_test.shape))
        print("Low fidelity y_lf_val shape: {}".format(y_lf_val.shape))
        print(f"concatenation order: {X_names}")
        print(f"concatenation order: {Y_lf_names}")
    
    print("Low fidelity data generated, scaled, and split. Low fidelity output data truncation: ", downsampleLF)

def genPredictionsForError(
    modelType:str, modelObject, fidelityLevel:str, verbose:bool,truncate:bool, X_test, X_train, y_test, y_train
    ): 
    modelName = modelType

    dataSplitNames = [
        '_' + modelName + '_test_predict',
        '_' + modelName + '_train_predict',
        '_' + fidelityLevel + '_test_truth',
        '_' + fidelityLevel + '_train_truth'
        ]
    numDataSplit = int(len(dataSplitNames)/2) # are we splitting into train/test (numDataSplit = 2) or train/test/validation (numDataSplit =3) ? 

    #######################################

    locals()[dataSplitNames[0][1:]] = modelObject.predict(X_test)
    locals()[dataSplitNames[1][1:]] = modelObject.predict(X_train)


    locals()[dataSplitNames[2][1:]] = y_test
    locals()[dataSplitNames[3][1:]] = y_train
    #######################################

    variableNameList =[] #to be used for the "un-truncating"
    if verbose:
        print(f'outputVarnames: {outputVarNames}, for i in np.arange(len(outputVarNames)): {np.arange(len(outputVarNames))}')
    for i in np.arange(len(outputVarNames)):
        for j in np.arange(numDataSplit):
            #Input Data Split
            locals()[outputVarNames[j] + dataSplitNames[i]] = np.hsplit(locals()[dataSplitNames[i][1:]],numDataSplit)[j]
            variableNameList.append(outputVarNames[j] + dataSplitNames[i])
            if verbose:
                print(f'{outputVarNames[j] + dataSplitNames[i]} = np.hsplit({[dataSplitNames[i][1:]]},{numDataSplit})[{j}]')
                print(f"variableNameList.append({outputVarNames[j] + dataSplitNames[i]})")

            locals()[outputVarNames[j] + dataSplitNames[i+2]] = np.hsplit(locals()[dataSplitNames[i+2][1:]],numDataSplit)[j]
            variableNameList.append(outputVarNames[j] + dataSplitNames[i+2])  
            if verbose:
                print(f'{outputVarNames[j] + dataSplitNames[i+2]} = np.hsplit({[dataSplitNames[i+2][1:]]},{numDataSplit})[{j}]')
                print(f"variableNameList.append({outputVarNames[j] + dataSplitNames[i+2]})") 
    time.sleep(2)
    if truncate: 
        leftFluidScalarDistributionLength = globals()[outputVarNames[0]][0,:xSpotLeft].shape[0]
        rightFluidScalarDistributionLength =  globals()[outputVarNames[0]][0,xSpotLeft:].shape[0]
        if verbose:
            print("leftFluidScalarDistributionLength:", leftFluidScalarDistributionLength)
            print("rightFluidScalarDistributionLength:", rightFluidScalarDistributionLength)

        for var in variableNameList:
            temp_left = [np.tile(entry, leftFluidScalarDistributionLength) for entry in locals()[var][:,0] ]
            temp_right = [np.tile(entry,rightFluidScalarDistributionLength) for entry in locals()[var][:,1] ]
            temp_stacked = np.hstack((temp_left,temp_right))
            locals()[var] = temp_stacked
            if verbose:
                print(var, " shape: ", locals()[var].shape)
########### Inverse Scale ###########

    if fidelityLevel == 'LF':

        for name in LFoutputVarNames:
            ScalerName = name + '_OutputScaler'
            for tail in dataSplitNames:
                fullName = name[0:-3] + tail
                locals()[fullName] = globals()[ScalerName].inverse_transform(locals()[fullName])
                if verbose:
                    print(fullName + ' = ' + ScalerName + '.inverse_transform(' +  fullName + ')')
                
    else:
        
        for name in outputVarNames:
            ScalerName = name + '_OutputScaler'
            for tail in dataSplitNames:
                fullName = name + tail
                locals()[fullName] = globals()[ScalerName].inverse_transform(locals()[fullName])
                if verbose:
                    print( fullName + ' has been inverse transformed using ' + ScalerName + '! It is called ' + fullName)

    single_nrmse = []
    single_R2 = []
    single_medianPeakMiss = []
    single_meanPeakMiss = []

    for var in outputVarNames:
        chosenVar = var
        chosenSet = 'test'
        matches = [match for match in variableNameList if (chosenSet in match) and (chosenVar+'_' in match)]

        truth = locals()[matches[1]]
        prediction = locals()[matches[0]]
        peakTruth = np.amax(truth,axis=1)
        peakPrediction = np.amax(prediction,axis=1)
        peakPercentDifference = percentdifferencecalc(peakTruth,peakPrediction)

        medianPeakMiss = np.median(peakPercentDifference)
        meanPeakMiss = np.mean(peakPercentDifference)
        rmse = mean_squared_error(truth, prediction, squared=False)
        ybar = truth.max()-truth.min()
        nrmse = round(rmse/ybar,5)
        R2 = round(r2_score(truth, prediction),4)
        
        single_R2.append(R2)
        single_nrmse.append(nrmse)
        single_medianPeakMiss.append(medianPeakMiss)
        single_meanPeakMiss.append(meanPeakMiss)

        locals()[matches[0]] = None
        locals()[matches[1]] = None


    average_nrmse = np.mean(single_nrmse)
    average_R2 = np.mean(single_R2)
    return single_nrmse, average_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss

def buildAndTrainModel(X_train, y_train,validData, fidelityLevel,modelType,truncate:bool,verbose:bool,frequencyData: bool, layerSizeList=None,kernel=None, n_restarts=None, hyperparamDict=None, callbacks_list=None):

    desiredXpredictSize = 1000
    multiplier = math.floor(desiredXpredictSize/X_train.shape[0])
    X_predictionSpeed = np.tile(X_train,(multiplier,1))
    frequencyTempList = []
    numTestCases = X_predictionSpeed.shape[0]

    (X_test, y_test) = validData 

    if modelType == 'NN':
        NN = None
        NN = build_model_parameterized(
                input_data = X_train, 
                output_data = y_train,
                layerSizeList = layerSizeList, 
                rate = hyperparamDict["learningRate"], 
                regType = hyperparamDict["regType"], 
                regValue = hyperparamDict["regValue"],
                hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"],
                outputLayerActivation = hyperparamDict["outputLayerActivation"],
                outputLayerDataType= 'float32',
                kernelInitializer = hyperparamDict["kernelInitializer"],
                optimizer = hyperparamDict["optimizer"],
                loss = hyperparamDict["loss"])

        NN_epochs = None
        NN_history = None
        
        start = time.time()
        NN_epochs, NN_history = train_model_all_fidelity(
            model = NN, 
            input_data = X_train, 
            output_data = y_train,
            numEpochs = hyperparamDict["numEpochs"], 
            myBatchSize = hyperparamDict["myBatchSize"],
            validData = validData,
            callbacks_list= callbacks_list)
        end = time.time()
        trainTime = round(end-start,4)
        tf.get_logger().setLevel('WARNING')
        currentLocation = os.getcwd()
        os.mkdir('temp')
        os.chdir('temp')
        NN.save(os.getcwd())
        modelSize = (get_dir_size('.'))

        os.chdir(currentLocation)
        shutil.rmtree('temp')
        
    
    if modelType == 'krig':
        krig = None
        krig = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=n_restarts)
        start = time.time()
        krig.fit(X_train, y_train)
        end = time.time()
        trainTime = round(end-start,4)
        optimizedKernel = krig.kernel_
        krigPickle = pickle.dumps(krig)
        modelSize = sys.getsizeof(krigPickle)
        krigPickle = None
        

    model = locals()[modelType]
    if frequencyData:
        frequencyTempList = []
        predictionTimeTempList = []
        numTestCases = X_predictionSpeed.shape[0]
        predictionLoopStart = time.time()
        for _ in np.arange(0,200):
            predictionStart = time.time()
            model.predict(X_predictionSpeed)
            predictionEnd = time.time()
            predictionTime = predictionEnd-predictionStart
            predictionTimeTempList.append(predictionTime)
            try:
                frequency = 1/((predictionTime)/numTestCases)
                frequencyTempList.append(frequency)
            except:
                print('frequency too fast!')
                pass
        predictionLoopEnd = time.time()
        frequencyTestAverage = np.mean(frequencyTempList)
        predictionLoopTime = round(predictionLoopEnd-predictionLoopStart,4)
        print(f'Train time: {trainTime}.  Prediction time: {predictionLoopTime}')
    else: 
        frequencyTestAverage = 'freq not tested'

    single_nrmse, average_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss = genPredictionsForError(
        modelType=modelType,
        modelObject=model,
        fidelityLevel=fidelityLevel,
        verbose=verbose,
        truncate=truncate,
        X_test=X_test,
        X_train=X_train,
        y_test=y_test,
        y_train= y_train
        )

    model = None; NN = None; krig = None

    if modelType == 'krig':
        return optimizedKernel, trainTime, average_nrmse, single_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss, modelSize, frequencyTestAverage
    elif modelType == 'NN':
        return NN_epochs, NN_history, trainTime, average_nrmse, single_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss, modelSize, frequencyTestAverage

def cleanKernelList(kernelList):
    kernelList = [str(item) for item in kernelList]

    disallowed_characters = '()'
    firstCleanList = []
    for s in kernelList:
        for char in disallowed_characters:
            s = s.replace(char, ' ')
        s = s.translate({ord(k): None for k in digits})
        firstCleanList.append(s)

    secondCleanList = []
    for element in firstCleanList:
        secondCleanList.append(element.split( ))

    cleanedKernelList = []
    for string in secondCleanList:
        cleanedKernelList.append([item for item in string if item[0].isupper()])

    return cleanedKernelList
def modelConvergenceStudy(
    fidelityLevel: str, modelType: str, M_inf, inputTrainingData, outputTrainingData, inputTrainingNames, outputTrainingNames, top_n_modelChoices: list, splitList: list, frequencyData: bool, verbose: bool, truncate: bool, hyperparamDict=None, callbacks_list=None, n_restarts=None, 
    ):

    #Set variables
    numIters = len(splitList)
    modelName = fidelityLevel + "_" + modelType
    dictName = modelName+"_modelConvergenceDict"
    polyFitDict = {"NN": 1, "krig":2}

    #Initialize data structures
    completionTime = []
    globals()[dictName]= dict()
    
    #Store list of model choices in dictionary
    globals()[dictName]["top_n_modelChoices"] = top_n_modelChoices

    #Create list of strings, top model choices. Used in dictionary
    annotations = [str(item) for item in top_n_modelChoices]

    # Name and create directories/paths
    currentConvStudyTopDir = modelName + time.strftime("%Y%m%d-%H%M%S")
    convStudyPath = os.path.join(path,modelConvStudyDir)
    os.chdir(convStudyPath)
    os.mkdir(currentConvStudyTopDir)
    os.chdir(currentConvStudyTopDir)

    ##### Store input and output data in X and y variables
    if fidelityLevel != 'MF':
        X = np.hstack(inputTrainingData)
    if fidelityLevel == 'MF':
        X = inputTrainingData
    y = np.hstack(outputTrainingData)
    Y_names = np.hstack(outputTrainingNames)
    X_names = np.hstack(inputTrainingNames)
    originalIdx = np.arange(0,X.shape[0])

    for i, split in enumerate(splitList):
        start = time.time()
        test_size = round(1-split,2)
        currentIter = i+1
        print(f'Training on {round(split*100,1)} percent of original train data. Data split # {currentIter} of {len(splitList)} begin. ')

        ###### Split Data
        X_train, X_test, y_train, y_test, M_inf_train, M_inf_test, trainIdx, testIdx = train_test_split(
            X, y, M_inf, originalIdx, test_size=test_size, random_state=random_state)

        X_test, X_val, y_test, y_val, M_inf_test, M_inf_val, testIdx, valIdx = train_test_split(
            X_test, y_test, M_inf_test, testIdx, test_size=0.50, random_state=random_state)

        validData = (X_test, y_test)

        if verbose:
            print("X_train shape: {}".format(X_train.shape))
            print("X_test shape: {}".format(X_test.shape))
            print("X_val shape: {}".format(X_val.shape))
            print("y_train shape: {}".format(y_train.shape))
            print("y_test shape: {}".format(y_test.shape))
            print("y_val shape: {}".format(y_val.shape))
            print(f"concatenation order: {X_names}")
            print(f"concatenation order: {Y_names}")

        ###### Build and Train Models ######

        ## Initialize Lists ##
        
        kernelList = []
        frequencyList = []
        nrmse_list = []
        trainTimeList = [] 
        modelSizeList = []
        single_nrmse_list = []
        R2_list = []
        single_R2_list = []
        single_medianPeakMissList = []
        single_meanPeakMissList = []

        for i, modelArchitecture in enumerate(top_n_modelChoices):
            currentInternalIter = i+1
            print(f'Model train iteration {currentIter}.{currentInternalIter} of {currentIter}.{len(top_n_modelChoices)} begin. (Total iters: {len(splitList)}) Model architecture: {str(modelArchitecture)}. ')

            tempDict = {}
            if modelType == 'NN':
                NN_epochs, NN_history, trainTime, average_nrmse, single_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss, modelSize, frequencyTestAverage = buildAndTrainModel(
                    X_train=X_train,
                    y_train=y_train,
                    fidelityLevel = fidelityLevel,
                    modelType = modelType,
                    layerSizeList=modelArchitecture,
                    hyperparamDict=hyperparamDict,
                    validData=validData,
                    callbacks_list = callbacks_list,
                    truncate = truncate,
                    verbose=verbose,
                    frequencyData = frequencyData
                    )
                tempDict.update( {
                'epochs' : NN_epochs,
                'history' : NN_history,
                } )

            elif modelType == 'krig':
                optimizedKernel, trainTime, average_nrmse, single_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss, modelSize, frequencyTestAverage = buildAndTrainModel(
                    X_train=X_train,
                    y_train=y_train,
                    fidelityLevel = fidelityLevel,
                    modelType = modelType,
                    kernel=modelArchitecture,
                    n_restarts=n_restarts,
                    validData=validData,
                    truncate = truncate,
                    verbose=verbose,
                    frequencyData = frequencyData
                    )
                kernelList.append(optimizedKernel)
                tempDict.update( {
                'originalKernel' : modelArchitecture,
                'optimizedKernel' : optimizedKernel,
                } )

            frequencyList.append(frequencyTestAverage)
            nrmse_list.append(average_nrmse)
            trainTimeList.append(trainTime)
            modelSizeList.append(modelSize)
            single_nrmse_list.append(single_nrmse)
            R2_list.append(average_R2)
            single_R2_list.append(single_R2)
            single_medianPeakMissList.append(single_medianPeakMiss)
            single_meanPeakMissList.append(single_meanPeakMiss)

            tempDict.update( {
                'average_nrmse' : average_nrmse,
                'single_nrmse' : single_nrmse,
                'trainTime' : trainTime,
                'modelSize' : modelSize,
                'frequency' : frequencyTestAverage,
                'single_R2': single_R2,
                'average_R2': average_R2,
                'single_medianPeakMiss' : single_medianPeakMiss,
                'single_meanPeakMiss' : single_meanPeakMiss
                } )

            globals()[dictName][split] = {
                annotations[i] : tempDict
            }

            print(f'Model train iteration {currentIter}.{currentInternalIter} of {currentIter}.{len(top_n_modelChoices)} complete. (Total iters: {len(splitList)})') 
            
            globals()[dictName][split].update( {
                'modelSizeList' : modelSizeList,
                'nrmse_list' : nrmse_list,
                'single_nrmse_list' : single_nrmse_list,
                'R2_list': R2_list,
                'single_R2_list': single_R2_list,
                'frequencyList' : frequencyList,
                'trainTimeList' : trainTimeList,
                'single_medianPeakMissList' : single_medianPeakMissList,
                'single_meanPeakMissList' : single_meanPeakMissList
            }
            )

        end = time.time()
        currentLoopTimeElapsed = round((end-start)/60,4)
        completionTime.append(currentLoopTimeElapsed)
        totalTime = round(np.sum(completionTime),4)
        # estimatedRemainingTime = round((numIters - currentIter)*np.mean(completionTime),4)

        print(f'Iteration {currentIter} of {numIters} complete. Loop time elapsed: {currentLoopTimeElapsed} minutes')
        polynomialOrder = polyFitDict[modelType]
        if currentIter != 1:
            iters = np.arange(0,currentIter)
            z = np.polyfit(iters,completionTime,polynomialOrder)
            polynomial = np.poly1d(z)
            xRemaining = np.arange(currentIter+1,numIters)
            yRemaining = polynomial(xRemaining)
            estimatedRemainingTime = round(np.sum(yRemaining),4)
        else: 
            estimatedRemainingTime = round((numIters - currentIter)*np.mean(completionTime),4)
        hoursElapsed = math.floor(totalTime / 60)
        minutesElapsed = math.floor(totalTime % 60)
        hoursRemain = math.floor(estimatedRemainingTime / 60)
        minutesRemain = math.floor(estimatedRemainingTime % 60)
        print(f'Total time elapsed: {hoursElapsed} hours & {minutesElapsed} minute(s). Estimated time remaining: {hoursRemain} hours & {minutesRemain} minute(s). ')

    #### Save dictionary 
    saveVersionedPickle(
    filename=dictName, 
    objectToSave=globals()[dictName],
    path = os.path.join(convStudyPath,currentConvStudyTopDir)
    )

    #### Plotting ####
    os.chdir(path)
    print(f'Dictionary name: {dictName}, saved at {str(os.path.join(convStudyPath,currentConvStudyTopDir))}')
    print('Convergence study complete. ')
    return globals()[dictName]

def percentileFilter(lowerPercentile,upperPercentile,inputVarNameList):
    # lowerBoundList = []
    # upperBoundList = []
    internalIdxList = []

    for inputVar in inputVarNameList:
        lowerBound = np.percentile(globals()[inputVar],lowerPercentile)
        upperBound = np.percentile(globals()[inputVar],upperPercentile)
        internalIdx = np.nonzero((globals()[inputVar].reshape(-1,)>lowerBound) & (globals()[inputVar].reshape(-1,)<upperBound))
        # upperBoundList.append(upperBound)
        # lowerBoundList.append(lowerBound)
        internalIdxList.append(internalIdx)

    commonIdx = reduce(
        np.intersect1d, internalIdxList
    )
    return commonIdx

def buildTrainModelForComparison(top_n_modelChoices: list, trainTestSplit:float, fidelityLevel: str, modelType: str, M_inf, inputTrainingData, outputTrainingData, inputTrainingNames, outputTrainingNames, verbose: bool, truncate: bool, frequencyData: bool, hyperparamDict=None, callbacks_list=None, n_restarts=None):

    #Set variables
    numIters = len(top_n_modelChoices)
    modelName = fidelityLevel + "_" + modelType
    dictName = modelName+"_modelArchitectureCompDict"

    #Initialize data structures
    completionTime = []
    globals()[dictName]= dict()
    
    #Store list of model choices in dictionary
    globals()[dictName]["top_n_modelChoices"] = top_n_modelChoices

    #Create list of strings, top model choices. Used in dictionary
    annotations = [str(item) for item in top_n_modelChoices]

    # Name and create directories/paths
    currentCompTopDir = modelName + time.strftime("%Y%m%d-%H%M%S")
    compStudyPath = os.path.join(path,modelCompDir)
    os.chdir(compStudyPath)
    os.mkdir(currentCompTopDir)
    os.chdir(currentCompTopDir)

    ##### Store input and output data in X and y variables
    if fidelityLevel != 'MF':
        X = np.hstack(inputTrainingData)
    if fidelityLevel == 'MF':
        X = inputTrainingData
    y = np.hstack(outputTrainingData)
    Y_names = np.hstack(outputTrainingNames)
    X_names = np.hstack(inputTrainingNames)
    originalIdx = np.arange(0,X.shape[0])

    test_size = round(1-trainTestSplit,2)
    if verbose:
        print(f'Training on {round(trainTestSplit*100,1)} percent of original train data.')

    ###### Split Data
    X_train, X_test, y_train, y_test, M_inf_train, M_inf_test, trainIdx, testIdx = train_test_split(
        X, y, M_inf, originalIdx, test_size=test_size, random_state=random_state)

    X_test, X_val, y_test, y_val, M_inf_test, M_inf_val, testIdx, valIdx = train_test_split(
        X_test, y_test, M_inf_test, testIdx, test_size=0.50, random_state=random_state)

    validData = (X_test, y_test)

    if verbose:
        print("X_train shape: {}".format(X_train.shape))
        print("X_test shape: {}".format(X_test.shape))
        print("X_val shape: {}".format(X_val.shape))
        print("y_train shape: {}".format(y_train.shape))
        print("y_test shape: {}".format(y_test.shape))
        print("y_val shape: {}".format(y_val.shape))
        print(f"concatenation order: {X_names}")
        print(f"concatenation order: {Y_names}")

    ###### Build and Train Models ######

    ## Initialize Lists ##
    
    kernelList = []
    frequencyList = []
    nrmse_list = []
    trainTimeList = [] 
    modelSizeList = []
    single_nrmse_list = []
    R2_list = []
    single_R2_list = []
    single_medianPeakMissList = []
    single_meanPeakMissList = []

    for i, modelArchitecture in enumerate(top_n_modelChoices):
        start = time.time()
        currentInternalIter = i+1
        print(f'Training model architecture {currentInternalIter} of {len(top_n_modelChoices)} begin. (Total iters: {len(top_n_modelChoices)}) Model architecture: {str(modelArchitecture)}. ')

        tempDict = {}
        if modelType == 'NN':
            NN_epochs, NN_history, trainTime, average_nrmse, single_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss, modelSize, frequencyTestAverage = buildAndTrainModel(
                X_train=X_train,
                y_train=y_train,
                fidelityLevel = fidelityLevel,
                modelType = modelType,
                layerSizeList=modelArchitecture,
                hyperparamDict=hyperparamDict,
                validData=validData,
                callbacks_list = callbacks_list,
                truncate = truncate,
                verbose=verbose,
                frequencyData = frequencyData
                )
            tempDict.update( {
            'epochs' : NN_epochs,
            'history' : NN_history,
            } )

        elif modelType == 'krig':
            optimizedKernel, trainTime, average_nrmse, single_nrmse, single_R2, average_R2, single_medianPeakMiss, single_meanPeakMiss, modelSize, frequencyTestAverage = buildAndTrainModel(
                X_train=X_train,
                y_train=y_train,
                fidelityLevel = fidelityLevel,
                modelType = modelType,
                kernel=modelArchitecture,
                n_restarts=n_restarts,
                validData=validData,
                truncate = truncate,
                verbose=verbose,
                frequencyData = frequencyData
                )
            kernelList.append(optimizedKernel)
            tempDict.update( {
            'originalKernel' : modelArchitecture,
            'optimizedKernel' : optimizedKernel,
            } )

        frequencyList.append(frequencyTestAverage)
        nrmse_list.append(average_nrmse)
        trainTimeList.append(trainTime)
        modelSizeList.append(modelSize)
        single_nrmse_list.append(single_nrmse)
        R2_list.append(average_R2)
        single_R2_list.append(single_R2)
        single_medianPeakMissList.append(single_medianPeakMiss)
        single_meanPeakMissList.append(single_meanPeakMiss)

        tempDict.update( {
            'average_nrmse' : average_nrmse,
            'single_nrmse' : single_nrmse,
            'trainTime' : trainTime,
            'modelSize' : modelSize,
            'frequency' : frequencyTestAverage,
            'single_R2': single_R2,
            'average_R2': average_R2,
            'single_medianPeakMiss' : single_medianPeakMiss,
            'single_meanPeakMiss' : single_meanPeakMiss
            } )

        globals()[dictName][split] = {
            annotations[i] : tempDict
        }

        print(f'Model train iteration {currentInternalIter} of {len(top_n_modelChoices)} complete. (Total iters: {len(top_n_modelChoices)})') 
        
        globals()[dictName][split].update( {
            'modelSizeList' : modelSizeList,
            'nrmse_list' : nrmse_list,
            'single_nrmse_list' : single_nrmse_list,
            'R2_list': R2_list,
            'single_R2_list': single_R2_list,
            'frequencyList' : frequencyList,
            'trainTimeList' : trainTimeList,
            'single_medianPeakMissList' : single_medianPeakMissList,
            'single_meanPeakMissList' : single_meanPeakMissList
        }
        )

        end = time.time()
        currentLoopTimeElapsed = round((end-start)/60,4)
        completionTime.append(currentLoopTimeElapsed)
        totalTime = round(np.sum(completionTime),4)
        MinutesGet, SecondsGet = divmod(totalTime, 60)
        HoursGet, MinutesGet = divmod(MinutesGet,60)
        estimatedRemainingTime = round((numIters - currentInternalIter)*np.mean(completionTime),4)
        MinutesLeft, SecondsLeft = divmod(estimatedRemainingTime, 60)
        HoursLeft, MinutesLeft = divmod(MinutesGet,60)
        print(f'Total time elapsed: {HoursGet} hour(s), {MinutesGet} minute(s), {SecondsGet} second(s). Estimated time remaining: {HoursLeft} hour(s), {MinutesLeft} minute(s), {SecondsLeft} second(s). ')

    #### Save dictionary 
    saveVersionedPickle(
    filename=dictName, 
    objectToSave=globals()[dictName],
    path = os.path.join(convStudyPath,currentConvStudyTopDir)
    )

    #### Plotting ####
    os.chdir(path)
    print(f'Dictionary name: {dictName}, saved at {str(os.path.join(convStudyPath,currentConvStudyTopDir))}')
    print('Convergence study complete. ')
    # you have like ten functions that do just this. use them directly or take their constiuent components to build this function.buildAndTrainModel or generateInverseTransformedPredictions or the model convergence function (but modified.). 
    return globals()[dictName], predictionList

def NNmemoryDuringTraining(X_train,y_train,hyperparamDict,dataMultiplierList):
    peakMemoryList = []
    for dataMultiplier in dataMultiplierList:
        print(f'Start loop data multiplier:{dataMultiplier}')
        tempMemorySaveListName = "tempList"
        globals()[tempMemorySaveListName] = []

        temp_X_train = np.tile(X_train,(int(dataMultiplier),1))
        temp_y_train = np.tile(y_train,(int(dataMultiplier),1))
        NN = build_model_parameterized(
            input_data = temp_X_train, 
            output_data = temp_y_train,
            layerSizeList = hyperparamDict["layerSize"], 
            rate = hyperparamDict["learningRate"], 
            regType = hyperparamDict["regType"], 
            regValue = hyperparamDict["regValue"],
            hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"],
            outputLayerActivation = hyperparamDict["outputLayerActivation"],
            outputLayerDataType= 'float32',
            kernelInitializer = hyperparamDict["kernelInitializer"],
            optimizer = hyperparamDict["optimizer"],
            loss = hyperparamDict["loss"]
            )

        class MemorySavingCallback(tf.keras.callbacks.Callback):
            def __init__(self,memorySaveListName):
                self.memorySaveListName = memorySaveListName
            def on_epoch_end(self, epoch, logs=None):
                gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
                #this is hackish. sorry to anyone who has the misfortune to see how I've done this. 
                globals()[self.memorySaveListName].append(float(gpu_dict['peak']) / (1024 ** 3))

        _ = train_model_all_fidelity(
            model = NN, 
            input_data = temp_X_train, 
            output_data = temp_y_train,
            numEpochs = hyperparamDict["numEpochs"], 
            myBatchSize = hyperparamDict["myBatchSize"],
            validData = (temp_X_train,temp_y_train),
            callbacks_list= [MemorySavingCallback(memorySaveListName= tempMemorySaveListName),MemoryPrintingCallback()]
            )
        peakMemoryList.append(np.max(globals()[tempMemorySaveListName]))
        print(f'End loop data multiplier:{dataMultiplier}. Peak memory: {np.max(globals()[tempMemorySaveListName])} gb')
    return peakMemoryList

def modelStorageCalculator(X_train,y_train,numLFoutputs,dataMultiplierList,hyperparamDict=None,kernel=None):
    modelParamsList = []
    trainDataParamsList = []
    modelSizeOnDiskList = []
    trainDataSizeOnDiskList = []
    trainTimeList = []

    numTrainInputPoints = X_train.shape[0]
    numTrainInputs = X_train.shape[1]
    numTrainOutputPoints = y_train.shape[0]
    numTrainOutputs = y_train.shape[1]
    numTrainOutputs_lf = numLFoutputs
    
    for i, dataMultiplier in enumerate(dataMultiplierList):
        print(f'Start loop data multiplier:{round(dataMultiplier,2)}')

        xVirtual_hf_shapeTuple = (int(dataMultiplier*numTrainInputPoints), numTrainInputs)
        xVirtual_lf_shapeTuple = (int(2*dataMultiplier*numTrainInputPoints), numTrainInputs)
        xVirtual_mf_shapeTuple = (int(dataMultiplier*numTrainInputPoints), numTrainInputs+numTrainOutputs_lf)
        xVirtual_hf = np.random.random(xVirtual_hf_shapeTuple)
        xVirtual_lf = np.random.random(xVirtual_lf_shapeTuple)
        xVirtual_mf = np.random.random(xVirtual_mf_shapeTuple)

        yVirtual_hf_shapeTuple = (int(dataMultiplier*numTrainOutputPoints), numTrainOutputs)
        yVirtual_lf_shapeTuple = (int(2*dataMultiplier*numTrainOutputPoints), numTrainOutputs_lf)
        yVirtual_mf_shapeTuple= (int(dataMultiplier*numTrainOutputPoints), numTrainOutputs) 
        yVirtual_hf = np.random.random(yVirtual_hf_shapeTuple)
        yVirtual_lf = np.random.random(yVirtual_lf_shapeTuple)
        yVirtual_mf = np.random.random(yVirtual_mf_shapeTuple)

        print(f'xVirtual_lf_shapeTuple: {xVirtual_lf_shapeTuple}, xVirtual_mf_shapeTuple: {xVirtual_mf_shapeTuple}')
        print(f'yVirtual_lf_shapeTuple: {yVirtual_lf_shapeTuple}, yVirtual_mf_shapeTuple: {yVirtual_mf_shapeTuple}')

        trainDataParamsList.append(yVirtual_hf.size)
        
        if hyperparamDict:
            # HF_NN = build_model_parameterized(
            #     input_data = xVirtual_hf, 
            #     output_data = yVirtual_hf,
            #     layerSizeList = hyperparamDict["layerSize"], 
            #     rate = hyperparamDict["learningRate"], 
            #     regType = hyperparamDict["regType"], 
            #     regValue = hyperparamDict["regValue"],
            #     hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"],
            #     outputLayerActivation = hyperparamDict["outputLayerActivation"],
            #     outputLayerDataType= 'float32',
            #     kernelInitializer = hyperparamDict["kernelInitializer"],
            #     optimizer = hyperparamDict["optimizer"],
            #     loss = hyperparamDict["loss"]
            #     )

            LF_NN = build_model_parameterized(
                input_data = xVirtual_lf, 
                output_data = yVirtual_lf,
                layerSizeList = hyperparamDict["layerSize"], 
                rate = hyperparamDict["learningRate"], 
                regType = hyperparamDict["regType"], 
                regValue = hyperparamDict["regValue"],
                hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"],
                outputLayerActivation = hyperparamDict["outputLayerActivation"],
                outputLayerDataType= 'float32',
                kernelInitializer = hyperparamDict["kernelInitializer"],
                optimizer = hyperparamDict["optimizer"],
                loss = hyperparamDict["loss"]
                )

            MF_NN = build_model_parameterized(
                input_data = xVirtual_mf, 
                output_data = yVirtual_mf,
                layerSizeList = hyperparamDict["layerSize"], 
                rate = hyperparamDict["learningRate"], 
                regType = hyperparamDict["regType"], 
                regValue = hyperparamDict["regValue"],
                hiddenLayerActivation = hyperparamDict["hiddenLayerActivation"],
                outputLayerActivation = hyperparamDict["outputLayerActivation"],
                outputLayerDataType= 'float32',
                kernelInitializer = hyperparamDict["kernelInitializer"],
                optimizer = hyperparamDict["optimizer"],
                loss = hyperparamDict["loss"]
                )
            modelParamsList.append(MF_NN.count_params())


            start = time.time()
            # #HF
            # _ = train_model_all_fidelity(
            #     model = HF_NN, 
            #     input_data = xVirtual_hf, 
            #     output_data = yVirtual_hf,
            #     numEpochs = hyperparamDict["numEpochs"], 
            #     myBatchSize = hyperparamDict["myBatchSize"],
            #     validData = (xVirtual_hf,yVirtual_hf),
            #     callbacks_list= None
            #     )
            #LF
            _ = train_model_all_fidelity(
                model = LF_NN, 
                input_data = xVirtual_lf, 
                output_data = yVirtual_lf,
                numEpochs = hyperparamDict["numEpochs"], 
                myBatchSize = hyperparamDict["myBatchSize"],
                validData = (xVirtual_lf,yVirtual_lf),
                callbacks_list= None
                )
            #MF
            _ = train_model_all_fidelity(
                model = MF_NN, 
                input_data = xVirtual_mf, 
                output_data = yVirtual_mf,
                numEpochs = hyperparamDict["numEpochs"], 
                myBatchSize = hyperparamDict["myBatchSize"],
                validData = (xVirtual_mf,yVirtual_mf),
                callbacks_list= None
                )
            end = time.time()
            trainTime = round(end-start,4)

            os.chdir(path)
            tempPath = path+ '\\temp'
            #Measure NN size on disk
            os.mkdir('temp')
            os.chdir('temp')
            os.mkdir('lf')
            os.chdir('lf')
            LF_NN.save(os.getcwd())
            os.chdir(tempPath)
            os.mkdir('mf')
            os.chdir('mf')
            MF_NN.save(os.getcwd())
            os.chdir(tempPath)
            modelSize = (get_dir_size('.'))
            os.chdir(path)
            shutil.rmtree('temp')
            modelSizeOnDiskList.append(modelSize)
        elif kernel:
            MF_krig = None
            LF_krig = None
            MF_krig = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
            LF_krig = gaussian_process.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
            start = time.time()
            LF_krig.fit(xVirtual_lf, yVirtual_lf)
            MF_krig.fit(xVirtual_mf, yVirtual_mf)
            end = time.time()
            trainTime = round(end-start,4)
            LFkrigPickle = pickle.dumps(LF_krig)
            MFkrigPickle = pickle.dumps(MF_krig)
            modelSize = sys.getsizeof(LFkrigPickle) + sys.getsizeof(MFkrigPickle)
            LFkrigPickle = None
            MFkrigPickle = None
            modelSizeOnDiskList.append(modelSize)

        trainTimeList.append(trainTime)

        #Measure training data size on disk. 
        trainDataPickle = pickle.dumps(yVirtual_hf)
        trainDataSize = sys.getsizeof(trainDataPickle)
        trainDataSizeOnDiskList.append(trainDataSize)
        trainDataPickle = None

        print(f'End loop data multiplier:{round(dataMultiplier,2)}. Iteration {i+1} of {len(dataMultiplierList)}')
    print('Done!')
    return modelParamsList, trainDataParamsList,trainDataSizeOnDiskList, modelSizeOnDiskList, trainTimeList