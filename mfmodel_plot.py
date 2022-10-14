def saveVectorizedFigures(name, print_pdf,print_svg):
    os.chdir(path)
    paperFiguresPath = os.path.normpath('./figures/AIAA_aviation')
    os.chdir(paperFiguresPath)
    if print_pdf:
        pdfName = name+'.pdf'
        fig.savefig(pdfName, format='pdf')
    if print_svg:
        svgName = name+'.svg'
        fig.savefig(svgName, format='svg')
    os.chdir(path)

def oneToOneVisualizationPlotAllData(
    case, qw_test_predict,p_test_predict, qw_test_truth, p_test_truth, M_inf_test, method
    ):

    plt.rcParams["figure.figsize"] = (15,10)
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=1.08, w_pad=1.5, h_pad=4)
    fig.patch.set_facecolor('white')

    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'colorMap' + '_' + method + '_' + dt

    elbowLocation = 2.35
    case = case
    cm = plt.cm.get_cmap('cool')
    zmax = 2.5
    z = np.arange(0,zmax, zmax/qw_test_predict[case,:].shape[0])
    #plot one case only
    labelstr = 'Mach inf: ' + str(M_inf_test[case]) + ',case:' + str(case)
    maxHeatTransfer = qw_test_predict[case,:].max()
    maxPressure = p_test_predict[case,:].max()

    ################### HEAT TRANSFER ########################
    x = qw_test_predict[case,:]/maxHeatTransfer
    y = qw_test_truth[case,:]/maxHeatTransfer
    sc = axs[0,0].scatter(x, y ,c = z, s=80, label = labelstr, 
                     cmap=cm,edgecolors='none',vmin=0,vmax=2.5 )
    cbar = fig.colorbar(sc,ax = axs[0,0])
    cbar.ax.set_title("x-location (meters)")
    cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w')
    axs[0,0].plot([0, 1], [0, 1], color = 'k')

    caseNRMSE = str(round(100*normalizedRootMeanSquaredError(qw_test_truth[case,:],qw_test_predict[case,:]),4))
    caseR2 =  str(round(r2_score(qw_test_truth[case,:], qw_test_predict[case,:]),4))
    plotTextBox = 'R2: ' + caseR2 + '\n' + 'NRMSE: ' + caseNRMSE + '%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0,0].text(0.05, 0.85, plotTextBox, transform=axs[0,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[0,0].set_title("Heat Transfer Predicitions vs Actual")
    axs[0,0].grid()
    axs[0,0].set_ylabel("True Value")
    axs[0,0].set_xlabel("Predicted Heat Transfer")

    #############################################

    axs1label = method + ' Prediction'
    sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[0,1].plot(x_cc_sorted[0,idxWindowStart:], qw_test_truth[case,:]/maxHeatTransfer, color='firebrick', 
                linestyle='solid', linewidth=4, label='Truth Data')

    axs[0,1].plot(x_cc_sorted[0,idxWindowStart:], qw_test_predict[case,:]/maxHeatTransfer, 
                color='black', linestyle='-.', linewidth=2, label=axs1label)
    # axs[0,1].set_title("Predicted Heat Transfer",fontsize='x-large')
    axs[0,1].set_ylabel("qw / qw_max", fontsize='x-large')
    axs[0,1].set_xlabel('x (meters)')

    axs[0,1].legend(fontsize='x-large')

    ############### PRESSURE ##################
    # 
    x = p_test_predict[case,:]/maxPressure
    y = p_test_truth[case,:]/maxPressure
    sc = axs[1,0].scatter(x, y ,c = z, s=80, label = labelstr, 
                     cmap=cm,edgecolors='none',vmin=0,vmax=2.5 )
    cbar = fig.colorbar(sc,ax = axs[1,0])
    cbar.ax.set_title("x-location (meters)")
    cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w')
    axs[1,0].plot([0, 1], [0, 1], color = 'k')

    caseNRMSE = str(round(100*normalizedRootMeanSquaredError(p_test_truth[case,:],p_test_predict[case,:]),4))
    caseR2 =  str(round(r2_score(p_test_truth[case,:], p_test_predict[case,:]),4))
    plotTextBox = 'R2: ' + caseR2 + '\n' + 'NRMSE: ' + caseNRMSE + '%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[1,0].text(0.05, 0.85, plotTextBox, transform=axs[1,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[1,0].set_title("Pressure Predicitions vs Actual")
    axs[1,0].grid()
    axs[1,0].set_ylabel("True Value")
    axs[1,0].set_xlabel("Predicted Pressure")

    #############################################

    axs1label = method + ' Prediction'
    sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[1,1].plot(x_cc_sorted[0,idxWindowStart:], p_test_truth[case,:]/maxPressure, color='firebrick', 
                linestyle='solid', linewidth=4, label='Truth Data')

    axs[1,1].plot(x_cc_sorted[0,idxWindowStart:], p_test_predict[case,:]/maxPressure, 
                color='black', linestyle='-.', linewidth=2, label=axs1label)
    # axs[1,1].set_title("Predicted Pressure",fontsize='x-large')
    axs[1,1].set_ylabel("p / p_max", fontsize='x-large')
    axs[1,1].set_xlabel('x (meters)')

    axs[1,1].legend(fontsize='x-large')


    os.chdir(figureDir)
    plt.savefig(figName)
    os.chdir(path)

def worstValidationErrorPlot(
    lowerPercentile, upperPercentile, testIdx, method, inputVarNameList, predictionList, architectureList
    ):

    plt.rcParams["figure.figsize"] = (12,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=1.08, w_pad=3, h_pad=4)
    fig.patch.set_facecolor('white')

    dt = time.strftime("%Y%m%d-%H%M%S")

    commonIdx = percentileFilter(
    lowerPercentile=lowerPercentile,
    upperPercentile=upperPercentile,
    inputVarNameList=inputVarNameList
    )

    filteredTestIdx = np.intersect1d(commonIdx, testIdx)
    indexList = []
    for index in filteredTestIdx:
        loc = np.where(testIdx == index)
        # print(loc)
        # print(f"Index {index} is at location: {loc[0]}. Verify {testIdx[loc[0]]} = {index}")
        indexList.append(loc[0])
    indexArray = np.asarray(indexList).reshape(-1,)
    nModels = np.shape(predictionList)[0]
    numColors = nModels
    colors = plt.cm.jet(np.linspace(0,1,numColors))

    
    # plt.scatter(qw_HF_test_truth[maxError,:]/maxHeatTransfer, qw_HF_krig_test_predict[maxError,:]/maxHeatTransfer)
    for i, list in enumerate(predictionList):
        qw_prediction = list[0]
        qw_truth = list[1]
        p_prediction = list[2]
        p_truth = list[3] 
        kernel = architectureList[i]
        currentColor = colors[i]
        currentLabel = f'{kernel} Prediction'

        errorList = []

        for idx in indexArray:
            p_nrmse = normalizedRootMeanSquaredError(p_truth[idx], p_prediction[idx])
            qw_nrmse = normalizedRootMeanSquaredError(qw_truth[idx], qw_prediction[idx])
            score = np.mean((p_nrmse, qw_nrmse))
            errorList.append(score)
        maxErrorCase = np.argmax(errorList)

        #plot one case only
        # labelstr = 'Mach inf: ' + str(M_inf_test[maxErrorCase]) + ',case:' + str(maxErrorCase)
        maxHeatTransfer = qw_truth[maxErrorCase,:].max()
        maxPressure = p_truth[maxErrorCase,:].max()

        #qw prediction
        axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw_prediction[maxErrorCase,:]/maxHeatTransfer, color=currentColor, linestyle='None', linewidth=.5, marker='D', markersize=5, mfc='white', markevery=20, label=currentLabel)

        #pressure predict
        axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_prediction[maxErrorCase,:]/maxPressure, color=currentColor, linestyle='None', linewidth=.5, marker='D', markersize=5, mfc='white', markevery=20, label=currentLabel)
    
    #qw truth
    axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw_truth[maxErrorCase,:]/maxHeatTransfer, color='black', linestyle='solid', linewidth=4, label='Truth Data', zorder=0)
    #pressure truth
    axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_truth[maxErrorCase,:]/maxPressure, color='black', linestyle='solid', linewidth=4, label='Truth Data', zorder=0)

    axs[0].set_xlabel("Wall Location, $x$ (meters)")
    axs[1].set_xlabel("Wall Location, $x$ (meters)")

    axs[0].set_ylabel("Scaled Heat Transfer Rate ($q_w/q_{w,max}$)")
    axs[1].set_ylabel("Scaled Pressure ($P/P_{max}$)")

    axs[0].legend()
    axs[1].legend()

    # ################### HEAT TRANSFER ########################
    # x = qw_test_predict[case,:]/maxHeatTransfer
    # y = qw_test_truth[case,:]/maxHeatTransfer
    # sc = axs[0,0].scatter(x, y ,c = z, s=80, label = labelstr, 
    #                  cmap=cm,edgecolors='none',vmin=0,vmax=2.5 )
    # cbar = fig.colorbar(sc,ax = axs[0,0])
    # cbar.ax.set_title("x-location (meters)")
    # cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w')
    # axs[0,0].plot([0, 1], [0, 1], color = 'k')

    # caseNRMSE = str(round(100*normalizedRootMeanSquaredError(qw_test_truth[case,:],qw_test_predict[case,:]),4))
    # caseR2 =  str(round(r2_score(qw_test_truth[case,:], qw_test_predict[case,:]),4))
    # plotTextBox = 'R2: ' + caseR2 + '\n' + 'NRMSE: ' + caseNRMSE + '%'

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # axs[0,0].text(0.05, 0.85, plotTextBox, transform=axs[0,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # # axs[0,0].set_title("Heat Transfer Predicitions vs Actual")
    # axs[0,0].grid()
    # axs[0,0].set_ylabel("True Value")
    # axs[0,0].set_xlabel("Predicted Heat Transfer")

    # #############################################

    # axs1label = method + ' Prediction'
    # sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    # # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    # axs[0,1].plot(x_cc_sorted[0,idxWindowStart:], qw_test_truth[case,:]/maxHeatTransfer, color='firebrick', 
    #             linestyle='solid', linewidth=4, label='Truth Data')

    # axs[0,1].plot(x_cc_sorted[0,idxWindowStart:], qw_test_predict[case,:]/maxHeatTransfer, 
    #             color='black', linestyle='-.', linewidth=2, label=axs1label)
    # # axs[0,1].set_title("Predicted Heat Transfer",fontsize='x-large')
    # axs[0,1].set_ylabel("qw / qw_max", fontsize='x-large')
    # axs[0,1].set_xlabel('x (meters)')

    # axs[0,1].legend(fontsize='x-large')

    # ############### PRESSURE ##################
    # # 
    # x = p_test_predict[case,:]/maxPressure
    # y = p_test_truth[case,:]/maxPressure
    # sc = axs[1,0].scatter(x, y ,c = z, s=80, label = labelstr, 
    #                  cmap=cm,edgecolors='none',vmin=0,vmax=2.5 )
    # cbar = fig.colorbar(sc,ax = axs[1,0])
    # cbar.ax.set_title("x-location (meters)")
    # cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w')
    # axs[1,0].plot([0, 1], [0, 1], color = 'k')

    # caseNRMSE = str(round(100*normalizedRootMeanSquaredError(p_test_truth[case,:],p_test_predict[case,:]),4))
    # caseR2 =  str(round(r2_score(p_test_truth[case,:], p_test_predict[case,:]),4))
    # plotTextBox = 'R2: ' + caseR2 + '\n' + 'NRMSE: ' + caseNRMSE + '%'

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # axs[1,0].text(0.05, 0.85, plotTextBox, transform=axs[1,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # # axs[1,0].set_title("Pressure Predicitions vs Actual")
    # axs[1,0].grid()
    # axs[1,0].set_ylabel("True Value")
    # axs[1,0].set_xlabel("Predicted Pressure")

    # #############################################

    # axs1label = method + ' Prediction'
    # sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

    # # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    # axs[1,1].plot(x_cc_sorted[0,idxWindowStart:], p_test_truth[case,:]/maxPressure, color='firebrick', 
    #             linestyle='solid', linewidth=4, label='Truth Data')

    # axs[1,1].plot(x_cc_sorted[0,idxWindowStart:], p_test_predict[case,:]/maxPressure, 
    #             color='black', linestyle='-.', linewidth=2, label=axs1label)
    # # axs[1,1].set_title("Predicted Pressure",fontsize='x-large')
    # axs[1,1].set_ylabel("p / p_max", fontsize='x-large')
    # axs[1,1].set_xlabel('x (meters)')

    # axs[1,1].legend(fontsize='x-large')

def oneToOnePlotTool(method, desiredNumCasesForPlot, X_test, qw_prediction, qw_truth, p_prediction, p_truth):

    totalCases = X_test.shape[0]
    casePlotRange= np.arange(0,totalCases,int((totalCases/desiredNumCasesForPlot)))

    plt.rcParams["figure.figsize"] = (10,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
    fig.patch.set_facecolor('white')

    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'oneToOneForManyCases' + '_' + method + '_' + dt

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
    os.chdir(path)
    os.chdir(figureDir)
    plt.savefig(figName)
    os.chdir(path)

def oneToOnePlotToolMultipleModels(method, desiredNumCasesForPlot, X_test, predictionList, architectureList):

    totalCases = X_test.shape[0]
    casePlotRange= np.arange(0,totalCases,int((totalCases/desiredNumCasesForPlot)))

    plt.rcParams["figure.figsize"] = (10,5)
    fig, axs = plt.subplots(1, 2)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=.5)
    fig.patch.set_facecolor('white')

    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'oneToOneForMultipleModels' + '_' + method + '_' + dt
    nModels = np.shape(predictionList)[0]
    numColors = nModels
    colors = plt.cm.jet(np.linspace(0,1,numColors))

    for i, list in enumerate(predictionList):
        qw_prediction = list[0]
        qw_truth = list[1]
        p_prediction = list[2]
        p_truth = list[3] 
        kernel = architectureList[i]
        currentColor = colors[i]
        currentLabel = f'Architecture: {kernel}'
        caseLoopCounter = 0
        for case in casePlotRange:
            if caseLoopCounter == 0:
                maxHeatTransfer = max(qw_prediction[case,:].max(),qw_truth[case,:].max())
                maxPressure = max(p_prediction[case,:].max(),p_truth[case,:].max())
                axs[0].scatter(qw_prediction[case,:]/maxHeatTransfer,qw_truth[case,:]/maxHeatTransfer, s=0.5,color=currentColor, label = currentLabel)
                axs[1].scatter(p_prediction[case,:]/maxPressure,p_truth[case,:]/maxPressure, s=0.5,color=currentColor, label = currentLabel)
            else:
                maxHeatTransfer = max(qw_prediction[case,:].max(),qw_truth[case,:].max())
                maxPressure = max(p_prediction[case,:].max(),p_truth[case,:].max())
                axs[0].scatter(qw_prediction[case,:]/maxHeatTransfer,qw_truth[case,:]/maxHeatTransfer, s=0.5,color=currentColor)
                axs[1].scatter(p_prediction[case,:]/maxPressure,p_truth[case,:]/maxPressure, s=0.5,color=currentColor)
            caseLoopCounter += 1

    # qwCaseNRMSE = str(round(100*normalizedRootMeanSquaredError(qw_truth,qw_prediction),4))
    # pCaseNRMSE = str(round(100*normalizedRootMeanSquaredError(p_truth,p_prediction),4))
    # qwCaseR2 =  str(round(r2_score(qw_truth, qw_prediction),4))
    # pCaseR2 =  str(round(r2_score(p_truth, p_prediction),4))
    # qwPlotTextBox = 'R2: ' + qwCaseR2 + '\n' + 'NRMSE: ' + qwCaseNRMSE + '%'
    # pPlotTextBox = 'R2: ' + pCaseR2 + '\n' + 'NRMSE: ' + pCaseNRMSE + '%'

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # axs[0].text(0.05, 0.85, qwPlotTextBox, transform=axs[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[1].text(0.05, 0.85, pPlotTextBox, transform=axs[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
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
    axs[0].legend(markerscale=10)
    axs[1].legend(markerscale=10)
    # os.chdir(path)
    # os.chdir(figureDir)
    # plt.savefig(figName)
    # os.chdir(path)

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
    axs[0].set_ylabel("$qw / qw_max$", fontsize='x-large')

    # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
    axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_test_truth[case,:]/maxPressure, color='black', linestyle='solid', linewidth=4, label='Truth Data')
    axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], p_test_predict[case,::sliceVal]/maxPressure, c='white',
                zorder=3,edgecolors='black', marker='D', s=70, label=method + ' Prediction')
    axs[1].set_title("Predicted Pressure", fontsize='x-large')
    axs[1].set_ylabel("$P/P_max$", fontsize='x-large')

    for i in np.arange(0,len(axs)):
        # axs[i].grid()
        axs[i].legend(fontsize='x-large')
        axs[i].set_xlabel('x (meters)',fontsize='x-large')
        # axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
        #     verticalalignment='top', bbox=props)

    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'pressureHeatTransferSideBySide' + '_' + method + '_' + dt
    os.chdir(path)
    os.chdir(figureDir)
    plt.savefig(figName)
    os.chdir(path)
        
def plotTrainAndTestLoss(historyDict,mseNames, colorList, fidelityLevel, epochRangeBegin=0,epochRangeEnd=None,):
    if epochRangeEnd == None:
        epochRangeEnd = len(historyDict['loss'])
    numValMSECount = sum(1 for string in mseNames if string.find('val') != -1)
    valMSEArray = np.zeros(((numValMSECount,epochRangeEnd)))
    minMSEArray = np.empty((numValMSECount,1))
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
    
    # summedValMSEArray = valMSEArray.sum(axis=0)

    plt.title(fidelityLevel + " Neural Network Loss")
    plt.legend(loc=0)
    plt.grid()
    # print('Average min epoch number: ' + str(np.mean(minMSEArray)))
    # print('Epoch number, minimum, all val error added: ' + str(np.argmin(summedValMSEArray)))
    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'trainAndTestLoss' + '_' + fidelityLevel + '_' + dt
    os.chdir(path)
    os.chdir(figureDir)
    plt.savefig(figName)
    os.chdir(path)

def plotAverageDistributions(qw_test_predict,p_test_predict, qw_test_truth, p_test_truth, method, fidelityLevel):
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
    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'averageDistribution' + '_' + fidelityLevel + '_' + method + '_' + dt
    os.chdir(path)
    os.chdir(figureDir)
    plt.savefig(figName)
    os.chdir(path)

def kerasPlotModel(model,fidelityLevel):
    os.chdir(path)
    os.chdir(figureDir)
    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'modelGraph' + '_' + fidelityLevel + '_' + dt
    try:
        tf.keras.utils.plot_model(
            model = model,
            to_file = figName,
            show_shapes=True,
            show_dtype=True,
            show_layer_activations=True
            )
        print('Successfully saved model graph via plot_model')
    except:
        print('Can\'t use \'plot model\', probably missing graphviz')

    os.chdir(path)

def plotPressureHeatTransferSideBySideTruthData(caseIndexArray, qw_truth, p_truth):
    plt.rcParams["figure.figsize"] = (15,5)
    fig, axs = plt.subplots(1, 2)
    fig.patch.set_facecolor('white')
    xStart = 0
    for case in caseIndexArray:
        # axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw_truth[case,:], linewidth=2, label='Case: ' + str(case))
        axs[0].semilogy(x_cc_windowed[0,xStart:], qw_truth[case,xStart:].reshape(-1,),label='Case: ' + str(case),linewidth=1)
        axs[0].set_title("Truth Heat Transfer",fontsize='x-large')
        axs[0].set_ylabel("qw", fontsize='x-large')

        # axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_truth[case,:], linewidth=2, label='Case: ' + str(case))
        axs[1].semilogy(x_cc_windowed[0,xStart:], p_truth[case,xStart:].reshape(-1,),label='Case: ' + str(case),linewidth=1)
        axs[1].set_title("Truth Pressure",fontsize='x-large')
        axs[1].set_ylabel("p", fontsize='x-large')

    for i in np.arange(0,len(axs)):
        axs[i].legend(fontsize='x-large')
        axs[i].set_xlabel('x (meters)',fontsize='x-large')

    dt = time.strftime("%Y%m%d-%H%M%S")
    figName = 'plotPressureHeatTransferSideBySideTruthData_' + dt
    os.chdir(path)
    os.chdir(figureDir)
    plt.savefig(figName)
    os.chdir(path)

def plotNNConvergence(fidelityLevel, numParamsList, MSElist, convStudyLayerList, savePlot):
    plt.rcParams["figure.figsize"] = (8,5)
    figName = 'NNconvergence' + fidelityLevel + '_'
    dt = str(datetime.date.today())
    figName += dt
    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(numParamsList, MSElist, label = 'data')
    if fidelityLevel == 'LF':
        outputVarNames_local = LFoutputVarNames
    else: 
        outputVarNames_local = outputVarNames

    totalParamsTrainData_local = 0
    for var in outputVarNames_local:
        totalParamsTrainData_local += globals()[var].shape[0] * globals()[var].shape[1]

    annotations = [str(item) for item in convStudyLayerList]

    for i, label in enumerate(annotations):
        ax.annotate(label, (numParamsList[i], MSElist[i]))

    ax.set_xlabel('Number of Neural Network Parameters')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_ylabel('Validation MSE')
    ax.set_title(fidelityLevel + ' Neural Network Convergence')

    def num2percentage(x):
        return (x / totalParamsTrainData_local)*100

    def percentage2num(x):
        return (x * totalParamsTrainData_local)/100

    secax = ax.secondary_xaxis('top', functions=(num2percentage, percentage2num))
    secax.set_xlabel('Train Data Params/Neural Network Params $(\%)$')

    # plt.show()
    if savePlot: 
        os.chdir(path)
        os.chdir(figureDir)
        baseName = figName
        counter = 2
        while os.path.exists('./' + figName + '.png'):
            figName = baseName + '_v' + str(counter)
            counter += 1
        plt.savefig(figName)
        os.chdir(path)
    plt.rcParams["figure.figsize"] = (6.4,4.8)

def plotInputSpace(caseList, inputVarNameList): #inputT, inputTw, inputU, inputRho):
# to use this function, the names in the inputVarNameList must already exist as a global variable
# this is sometimes called an All-against-all scatter plot
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Parameter Space Search", fontsize=18, y=0.95)

    colorList = ['red', 'grey']
    labelList = ['outliers', 'not outliers']
    zOrderList = [3,0 ]
        
    combinations = [list(item) for item in itertools.combinations(inputVarNameList, 2)]
    matches=[]
    for match in combinations:
        if not ("inputMach" in match and "inputTemperature" in match):
            if not ("inputMach" in match and "inputVelocity" in match):
                matches.append(match)
    combinations=matches

    for i, combo in enumerate(combinations):
        for j, caseArray in enumerate(caseList):
            ax = plt.subplot(len(combinations)//2,2, i+1)
            ax.scatter(globals()[combo[0]][caseArray], globals()[combo[1]][caseArray], c=colorList[j],label=labelList[j],zorder=zOrderList[j])
            if j == 0:
                ax.scatter(globals()[combo[0]][389], globals()[combo[1]][389], c='purple',label='case 389', zorder=zOrderList[j])
            ax.set_title(combo[0] + ' vs. ' + combo[1])
            ax.legend()
            ax.set_xlabel(combo[0])
            ax.set_ylabel(combo[1])

    # dt = time.strftime("%Y%m%d-%H%M%S")
    # figName = 'plotPressureHeatTransferSideBySideTruthData_' + dt
    # os.chdir(path)
    # os.chdir(figureDir)
    # plt.savefig(figName)
    # os.chdir(path)

def plotInputSpaceErrorColorMap(testIdxArray, trainIdxArray, inputVarNameList, topErrorValuesToPlot, errorList, logPlot): #inputT, inputTw, inputU, inputRho):
# to use this function, the names in the inputVarNameList must already exist as a global variable
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Parameter Space Search", fontsize=18, y=0.95)

    halfwayPoint = int(qw.shape[1]/2)
    qwChopped = qw[:,:halfwayPoint]
    qwMin = np.amin(qwChopped, axis=1)
    negativeQwMin = np.nonzero(qwMin<0)
    # weirdoCase = [196, 389]

    n = topErrorValuesToPlot
    topErrorListSortedIndex = np.argsort(-1*np.asarray(errorList))[:n]
    m = 10
    sortedList = np.sort(errorList)[::-1][:m]
    print('Top ', str(m), ' Error Values: ', sortedList)

    edgeColors = ['blue', 'purple', 'green', 'indigo']
    # colorList = ['red', 'grey']
    # labelList = ['outliers', 'not outliers']
    # zOrderList = [3,0 ]
    cm = plt.cm.get_cmap('autumn')
    cm = cm.reversed()
    combinations = [list(item) for item in itertools.combinations(inputVarNameList, 2)]
    matches=[]
    for match in combinations:
        if not ("inputMach" in match and "inputTemperature" in match):
            if not ("inputMach" in match and "inputVelocity" in match):
                matches.append(match)
    combinations=matches


    for i, combo in enumerate(combinations):
        # for j, caseArray in enumerate(caseList): #test, then train


        x = globals()[combo[0]][testIdxArray]
        y = globals()[combo[1]][testIdxArray]
        sc = ax.scatter( x, y ,c = z, cmap=cm,edgecolors='none', zorder = 3, s=60,marker='d',label='Test Data', norm=norm)
        if 'weirdoCase' in locals():
            for j, case in enumerate(weirdoCase):
                ax.scatter(globals()[combo[0]][case], globals()[combo[1]][case],s=90, edgecolors= edgeColors[j], facecolors='none',label='Case: ' + str(case), zorder=3)
        if 'topErrorListSortedIndex' in locals():
            # for j, case in enumerate(topErrorListSortedIndex):
            ax.scatter(globals()[combo[0]][testIdxArray][topErrorListSortedIndex], globals()[combo[1]][testIdxArray][topErrorListSortedIndex],s=80, edgecolors= 'black', facecolors='none',label='Top '+ str(n) + ' error, Cases: ' + str(topErrorListSortedIndex), zorder=3)
        # ax.scatter(globals()[combo[0]][negativeQwMin], globals()[combo[1]][negativeQwMin],s=80, edgecolors = 'black', facecolors='none',label='negative q_w', zorder=1)
        ax.set_title(combo[0] + ' vs. ' + combo[1])
        if i == 0:
            ax.legend(bbox_to_anchor=(-0.1, 1.7), loc='upper left', borderaxespad=0, facecolor='lightgrey')
        # ax.legend()
        ax.set_xlabel(combo[0])
        ax.set_ylabel(combo[1])
        cbar = plt.colorbar(sc,ax = ax)
        cbar.ax.set_title("$NRMSE$")
        ### PLot the mean R2 on the colorbar    # cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w'

def plotInputSpaceErrorColorMapLF(badBoysArray, LFinputVarNames): #inputT, inputTw, inputU, inputRho):
# to use this function, the names in the inputVarNameList must already exist as a global variable
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Parameter Space Search", fontsize=18, y=0.95)

    edgeColors = ['blue', 'purple', 'green', 'indigo', 'salmon']

    combinations = [list(item) for item in itertools.combinations(LFinputVarNames, 2)]
    # matches=[]
    # for match in combinations:
    #     if not ("inputMach" in match and "inputTemperature" in match):
    #         if not ("inputMach" in match and "inputVelocity" in match):
    #             matches.append(match)
    # combinations=matches
    print(combinations)

    for i, combo in enumerate(combinations):
        # for j, caseArray in enumerate(caseList): #test, then train
        ax = plt.subplot(len(combinations)//2,2, i+1)
        # z = errorList

        ax.scatter(globals()[combo[0]][:], globals()[combo[1]][:], c='lightgrey',label='Train Data',zorder=1, s=3)

        left = np.percentile(globals()[combo[0]],10)
        right = np.percentile(globals()[combo[0]],90)
        bottom = np.percentile(globals()[combo[1]],10)
        top = np.percentile(globals()[combo[1]],90)

        width = right-left
        height = top-bottom

        ax.add_patch(Rectangle((left, bottom), width, height, zorder=0,alpha=0.2))

        # x = globals()[combo[0]][testIdxArray]
        # y = globals()[combo[1]][testIdxArray]
        # sc = ax.scatter( x, y ,c = z, s=80, cmap=cm,edgecolors='none', zorder = 3, marker='d',label='Test Data')
        for j, case in enumerate(badBoysArray):
            ax.scatter(globals()[combo[0]][case], globals()[combo[1]][case],s=80, edgecolors= edgeColors[j], facecolors='none',label='Case: ' + str(case), zorder=3)
        # ax.scatter(globals()[combo[0]][negativeQwMin], globals()[combo[1]][negativeQwMin],s=80, edgecolors = 'black', facecolors='none',label='negative q_w', zorder=1)
        ax.set_title(combo[0] + ' vs. ' + combo[1])
        if i == 0:
            ax.legend(bbox_to_anchor=(-0.1, 1.5), loc='upper left', borderaxespad=0)
        # ax.legend()
        ax.set_xlabel(combo[0])
        ax.set_ylabel(combo[1])
        # cbar = plt.colorbar(sc,ax = ax)
        # cbar.ax.set_title("$RMSE$")
        ### PLot the mean R2 on the colorbar    # cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w'

def plotInputSpaceMissedPeakColorMap(testIdxArray, trainIdxArray, inputVarNameList, topErrorValuesToPlot, errorList, logPlot): #inputT, inputTw, inputU, inputRho):
# to use this function, the names in the inputVarNameList must already exist as a global variable
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Parameter Space Search", fontsize=18, y=0.95)

    halfwayPoint = int(qw.shape[1]/2)
    qwChopped = qw[:,:halfwayPoint]
    qwMin = np.amin(qwChopped, axis=1)
    negativeQwMin = np.nonzero(qwMin<0)
    # weirdoCase = [196, 389]

    n = topErrorValuesToPlot
    topErrorListSortedIndex = np.argsort(-1*np.asarray(errorList))[:n]
    m = 10
    sortedList = np.sort(errorList)[::-1][:m]
    print('Top ', str(m), ' Error Values: ', sortedList)

    edgeColors = ['blue', 'purple', 'green', 'indigo']
    # colorList = ['red', 'grey']
    # labelList = ['outliers', 'not outliers']
    # zOrderList = [3,0 ]
    cm = plt.cm.get_cmap('autumn')
    cm = cm.reversed()
    combinations = [list(item) for item in itertools.combinations(inputVarNameList, 2)]
    matches=[]
    for match in combinations:
        if not ("inputMach" in match and "inputTemperature" in match):
            if not ("inputMach" in match and "inputVelocity" in match):
                matches.append(match)
    combinations=matches


    for i, combo in enumerate(combinations):
        # for j, caseArray in enumerate(caseList): #test, then train
        ax = plt.subplot(len(combinations)//2,2, i+1)
        z = errorList
        if logPlot: 
            norm = matplotlib.colors.LogNorm()
        else: 
            norm = matplotlib.colors.Normalize()

        ax.scatter(globals()[combo[0]][trainIdxArray], globals()[combo[1]][trainIdxArray], c='lightgrey',label='Train Data',zorder=1, s=5)

        left = np.percentile(globals()[combo[0]],10)
        right = np.percentile(globals()[combo[0]],90)
        bottom = np.percentile(globals()[combo[1]],10)
        top = np.percentile(globals()[combo[1]],90)

        width = right-left
        height = top-bottom

        ax.add_patch(Rectangle((left, bottom), width, height, zorder=0,alpha=0.2))

        x = globals()[combo[0]][testIdxArray]
        y = globals()[combo[1]][testIdxArray]
        sc = ax.scatter( x, y ,c = z, cmap=cm,edgecolors='none', zorder = 3, s=60,marker='d',label='Test Data', norm=norm)
        if 'weirdoCase' in locals():
            for j, case in enumerate(weirdoCase):
                ax.scatter(globals()[combo[0]][case], globals()[combo[1]][case],s=90, edgecolors= edgeColors[j], facecolors='none',label='Case: ' + str(case), zorder=3)
        if 'topErrorListSortedIndex' in locals():
            # for j, case in enumerate(topErrorListSortedIndex):
            ax.scatter(globals()[combo[0]][testIdxArray][topErrorListSortedIndex], globals()[combo[1]][testIdxArray][topErrorListSortedIndex],s=80, edgecolors= 'black', facecolors='none',label='Top '+ str(n) + ' error, Cases: ' + str(topErrorListSortedIndex), zorder=3)
        # ax.scatter(globals()[combo[0]][negativeQwMin], globals()[combo[1]][negativeQwMin],s=80, edgecolors = 'black', facecolors='none',label='negative q_w', zorder=1)
        ax.set_title(combo[0] + ' vs. ' + combo[1])
        if i == 0:
            ax.legend(bbox_to_anchor=(-0.1, 1.7), loc='upper left', borderaxespad=0, facecolor='lightgrey')
        # ax.legend()
        ax.set_xlabel(combo[0])
        ax.set_ylabel(combo[1])
        cbar = plt.colorbar(sc,ax = ax)
        cbar.ax.set_title("$Peak Abs Err$")
        ### PLot the mean R2 on the colorbar    # cbar.ax.plot([0, zmax], [elbowLocation]*2, 'w'

def plotModelConvergenceStudy(top_n_modelChoices,splitList,convDict, fidelityLevel, modelChoice, peakMiss):
    markerList = [
        'o',
        'v',
        's',
        "*",
        'X',
        'p',
        '1',
        '>',
        'H',
        '4',
        'P'
    ]

    inputDict = {
        'R^2' : 'R2_list',
        'NRMSE' : 'nrmse_list', 
    }

    peakMissErrorDict = {
        'median': 'single_medianPeakMissList',
        'mean' : 'single_meanPeakMissList',
    }
    errorMetric = peakMissErrorDict[peakMiss]

    for key in inputDict.keys():
        yAxis = inputDict[key]
        nModels = len(top_n_modelChoices)
        numColors = nModels
        colors = plt.cm.jet(np.linspace(0,1,numColors))
        if modelChoice == 'krig':
            modelArchitectureList = cleanKernelList(kernelList = convDict['top_n_modelChoices'])
        if modelChoice == 'NN':
            modelArchitectureList = [str(item) for item in convDict['top_n_modelChoices']]

        plt.rcParams["figure.figsize"] = (8,5)
        figName = f'modelConvergence_{fidelityLevel}_{modelChoice}_'
        dt = str(datetime.date.today())
        figName += dt
        fig, ax = plt.subplots(constrained_layout=True)

        xList = [split*100 for split in splitList]
        yList = []
        labels = []
        for j, modelChoice in enumerate(np.arange(0,nModels)):
            labels.append(f'Model: {modelArchitectureList[modelChoice]}')
            for split in splitList:
                yList.append((convDict[split][yAxis][j]))

        iterator = int(len(xList))
        start = 0
        end = iterator
        for i in np.arange(0,nModels):
            ax.plot(xList,yList[start:end], linestyle='--',c=colors[i],marker=markerList[i], linewidth=0.75,zorder=0, label = labels[i])
            start += iterator
            end += iterator

        ax.set_xlabel('Training Data Split (%)')
        ax.set_ylabel(f'${key}$')
        ax.legend()
    
    peakMissList = []
    for split in splitList:
        peakMissList.append(convDict[split][errorMetric])
    peakMissArray = np.asarray(peakMissList)

    varDict = {
        'qw' : 'Heat Flux',
        'qw_LF' : 'Heat Flux',
        'p' : 'Pressure',
        'p_LF': 'Pressure'
    }

    for j, var in enumerate(outputVarNames):
        fig, ax = plt.subplots(constrained_layout=True)
        yList = peakMissArray[:,:,j]
        for i in np.arange(0,nModels):
            ax.plot(xList, yList[:,i], linestyle='--',c=colors[i],marker=markerList[i], linewidth=0.75,zorder=0, label = labels[i])
        ax.set_xlabel('Training Data Split (%)')
        ax.set_ylabel(f'{peakMiss.capitalize()} {varDict[var]} \n Peak To Peak Percent Difference (%)', wrap=True)
        ax.legend()

def plotModelConvergenceStudy_allModels(modelTypeList, top_n_modelChoicesList,splitList,convDict, peakMiss):
    markerList = [
        'o',
        'v',
        's',
        "*",
        'X',
        'p',
        '1',
        '>',
        'H',
        '4',
        'P'
    ]

    inputDict = {
        'R^2' : 'R2_list',
        'NRMSE' : 'nrmse_list', 
    }

    peakMissErrorDict = {
        'median': 'single_medianPeakMissList',
        'mean' : 'single_meanPeakMissList',
    }
    errorMetric = peakMissErrorDict[peakMiss]

    nModels = len(modelTypeList)
    numColors = nModels
    colors = plt.cm.jet(np.linspace(0,1,numColors))
    
    for model in modelTypeList:
        currentModelDict = convDict[model]

        for key in inputDict.keys():
            yAxis = inputDict[key]
            nModels = len(top_n_modelChoices)
            numColors = nModels
            colors = plt.cm.jet(np.linspace(0,1,numColors))
            if modelChoice == 'krig':
                modelArchitectureList = cleanKernelList(kernelList = convDict['top_n_modelChoices'])
            if modelChoice == 'NN':
                modelArchitectureList = [str(item) for item in convDict['top_n_modelChoices']]

            plt.rcParams["figure.figsize"] = (8,5)
            figName = f'modelConvergence_{fidelityLevel}_{modelChoice}_'
            dt = str(datetime.date.today())
            figName += dt
            fig, ax = plt.subplots(constrained_layout=True)

            xList = [split*100 for split in splitList]
            yList = []
            labels = []
            for j, modelChoice in enumerate(np.arange(0,nModels)):
                labels.append(f'Model: {modelArchitectureList[modelChoice]}')
                for split in splitList:
                    yList.append((convDict[split][yAxis][j]))

            iterator = int(len(xList))
            start = 0
            end = iterator
            for i in np.arange(0,nModels):
                ax.plot(xList,yList[start:end], linestyle='--',c=colors[i],marker=markerList[i], linewidth=0.75,zorder=0, label = labels[i])
                start += iterator
                end += iterator

            ax.set_xlabel('Training Data Split (%)')
            ax.set_ylabel(f'${key}$')
            ax.legend()
    
    # peakMissList = []
    # for split in splitList:
    #     peakMissList.append(convDict[split][errorMetric])
    # peakMissArray = np.asarray(peakMissList)

    # varDict = {
    #     'qw' : 'Heat Flux',
    #     'qw_LF' : 'Heat Flux',
    #     'p' : 'Pressure',
    #     'p_LF': 'Pressure'
    # }

    # for j, var in enumerate(outputVarNames):
    #     fig, ax = plt.subplots(constrained_layout=True)
    #     yList = peakMissArray[:,:,j]
    #     for i in np.arange(0,nModels):
    #         ax.plot(xList, yList[:,i], linestyle='--',c=colors[i],marker=markerList[i], linewidth=0.75,zorder=0, label = labels[i])
    #     ax.set_xlabel('Training Data Split (%)')
    #     ax.set_ylabel(f'{peakMiss.capitalize()} {varDict[var]} \n Peak To Peak Percent Difference (%)', wrap=True)
    #     ax.legend()

# def plotFluidScalarFidelityComparison(case, highFidelityFluidScalarList, multiFidelityFluidScalarList, method):
#     fluidScalarNames = [
#     'qw_test_predict','p_test_predict', 'qw_test_truth', 'p_test_truth'
#     ]

#     maxHeatTransferList = []
#     maxPressureList = []
    
#     for i, data in enumerate(highFidelityFluidScalarList):
#         locals()['HF' + fluidScalarNames[i]] = data
#         if "qw_" in fluidScalarNames[i]:
#             maxHeatTransferList.append(data[case,:].max())
#         if "p_" in fluidScalarNames[i]:
#             maxPressureList.append(data[case,:].max())

#     for i, data in enumerate(multiFidelityFluidScalarList):
#         locals()['MF' + fluidScalarNames[i]] = data
#         if "qw_" in fluidScalarNames[i]:
#             maxHeatTransferList.append(data[case,:].max())
#         if "p_" in fluidScalarNames[i]:
#             maxPressureList.append(data[case,:].max())

#     sliceVal = 20 # this is the "ol' fashioned way" for the plt.plot argument "markevery=sliceVal." The command doesn't work in plt.scatter

#     plt.rcParams["figure.figsize"] = (15,5)
#     fig, axs = plt.subplots(1, 2)
#     fig.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.5)
#     fig.patch.set_facecolor('white')


#     maxHeatTransfer = max(maxHeatTransferList)
#     maxPressure = max(maxPressureList)

#     # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
#     axs[0].plot(x_cc_sorted[0,idxWindowStart:], qw_test_truth[case,:]/maxHeatTransfer, color='firebrick', linestyle='solid', linewidth=4, label='Truth Data')
#     axs[0].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], qw_test_predict[case,::sliceVal]/maxHeatTransfer, c='white',
#                 zorder=3,edgecolors='black', marker='D', s=70, label=method + ' Prediction')
#     axs[0].set_title("Predicted Heat Transfer",fontsize='x-large')
#     axs[0].set_ylabel("qw / qw_max", fontsize='x-large')

#     # plt.plot(theta_rbf, Tw_rbf, color='black', linestyle='solid', linewidth=2, marker='D', markersize=6,     mfc='white', markevery=5, label='RBF')
#     axs[1].plot(x_cc_sorted[0,idxWindowStart:], p_test_truth[case,:]/maxPressure, color='black', linestyle='solid', linewidth=4, label='Truth Data')
#     axs[1].scatter(x_cc_sorted[0,idxWindowStart::sliceVal], p_test_predict[case,::sliceVal]/maxPressure, c='white',
#                 zorder=3,edgecolors='black', marker='D', s=70, label=method + ' Prediction')
#     axs[1].set_title("Predicted Pressure", fontsize='x-large')
#     axs[1].set_ylabel("P/P_max", fontsize='x-large')

#     for i in np.arange(0,len(axs)):
#         # axs[i].grid()
#         axs[i].legend(fontsize='x-large')
#         axs[i].set_xlabel('x (meters)',fontsize='x-large')
#         # axs[i].text(0.05, 0.55, textstr, transform=axs[i].transAxes, fontsize=14,
#         #     verticalalignment='top', bbox=props)

#     dt = time.strftime("%Y%m%d-%H%M%S")
#     figName = 'pressureHeatTransferSideBySide' + '_' + method + '_' + dt
#     os.chdir(path)
#     os.chdir(figureDir)
#     plt.savefig(figName)
#     os.chdir(path)