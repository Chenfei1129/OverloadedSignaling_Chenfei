import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def getProportionTargetReached(df):
    propTrialsTargetReached = pd.DataFrame(columns=['signaler',  'receiver', 'quit'])#, 'signaler_1word', 'signaler_2word'
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0
    
    getPropSignalerQuits =  lambda colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0

    sToGoal = [x for x in df.columns if 'sAchievesGoal' in x]
    #rToGoal1 = [x for x in df.columns if 'rAchievesGoal' in x and 'signal_length_1word' in x]
    #rToGoal2 = [x for x in df.columns if 'rAchievesGoal' in x and 'signal_length_2word' in x]
    rToGoal = [x for x in df.columns if 'rAchievesGoal' in x and 'signal_length_2word' not in x and 'signal_length_1word' not in x]
    #print(rToGoal)
    sAction = [x for x in df.columns if 'sChoice' in x]
    
    for sToGoalColumn, rToGoalColumn, signalerChoice in zip(sToGoal, rToGoal, sAction):
        
        colName = sToGoalColumn#re.findall(r"[A-Z_]+", sToGoalColumn)[0]
        propTrialsTargetReached.loc[colName] = [getPropTrue(sToGoalColumn), getPropTrue(rToGoalColumn), getPropSignalerQuits(signalerChoice)]
        #print(propTrialsTargetReached)
    #print(propTrialsTargetReached['receiver_word1'])
    #print(propTrialsTargetReached['receiver_word2'])
    #print(propTrialsTargetReached['receiver'])
    propTrialsTargetReached['total'] = propTrialsTargetReached['signaler'] + propTrialsTargetReached['receiver']#propTrialsTargetReached['receiver_word1'] + propTrialsTargetReached['receiver_word2'] #
    
    propTrialsTargetReached['receiver failure'] = 1-(propTrialsTargetReached['total']+propTrialsTargetReached['quit'])

    propTrialsTargetReached['marginOfErrorS'] =  1.96*np.sqrt((propTrialsTargetReached['signaler']*(1-propTrialsTargetReached['signaler']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorQ'] =  1.96*np.sqrt((propTrialsTargetReached['quit']*(1-propTrialsTargetReached['quit']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorRF'] =  1.96*np.sqrt((propTrialsTargetReached['receiver failure']*(1-propTrialsTargetReached['receiver failure']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorT'] =  1.96*np.sqrt((propTrialsTargetReached['total']*(1-propTrialsTargetReached['total']))/df.shape[0])
    print(propTrialsTargetReached)

    return(propTrialsTargetReached)

def getProportionTargetReachedWord(df):
    propTrialsTargetReached = pd.DataFrame(columns=['signaler_1word', 'signaler_2word', 'receiver', 'quit'])
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerQuits =  lambda colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0

    sToGoal = [x for x in df.columns if 'sAchievesGoal' in x]
    rToGoal = [x for x in df.columns if 'rAchievesGoal' in x] 
    sAction = [x for x in df.columns if 'sChoice' in x]
    
    for sToGoalColumn, rToGoalColumn, signalerChoice in zip(sToGoal, rToGoal, sAction):
        colName = sToGoalColumn#re.findall(r"[A-Z_]+", sToGoalColumn)[0]
        propTrialsTargetReached.loc[colName] = [getPropTrue(sToGoalColumn), getPropTrue(rToGoalColumn), getPropSignalerQuits(signalerChoice)]
        print(propTrialsTargetReached.loc[colName])
    propTrialsTargetReached['total'] = propTrialsTargetReached['signaler'] + propTrialsTargetReached['receiver']
    propTrialsTargetReached['receiver failure'] = 1-(propTrialsTargetReached['total']+propTrialsTargetReached['quit'])

    propTrialsTargetReached['marginOfErrorS'] =  1.96*np.sqrt((propTrialsTargetReached['signaler']*(1-propTrialsTargetReached['signaler']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorQ'] =  1.96*np.sqrt((propTrialsTargetReached['quit']*(1-propTrialsTargetReached['quit']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorRF'] =  1.96*np.sqrt((propTrialsTargetReached['receiver failure']*(1-propTrialsTargetReached['receiver failure']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorT'] =  1.96*np.sqrt((propTrialsTargetReached['total']*(1-propTrialsTargetReached['total']))/df.shape[0])

    return(propTrialsTargetReached)

def getPercentFromOptimalUtilityDF(df):
    percentFromOptimalUtil = pd.DataFrame(index=df.index)
    utilityColumns = [x for x in df.columns if 'utility' in x]
    
    for utilityName in utilityColumns:
        u_optimal = 'CentralControl_utility'
        percentFromOptimalUtil[utilityName] = 100.0*(df[utilityName]/df[u_optimal])#df[utilityName]-df[u_optimal] #
    return(percentFromOptimalUtil)


def getUtilityDifferenceSummaryStatistics(df):
    print("***")
    summaryDF = pd.DataFrame(df.mean(axis=0)).rename(columns={0:'means'})
    summaryDF['stds'] = df.std(axis=0)
    numObs = df.shape[0]
    summaryDF['marginOfError'] = 1.96*summaryDF['stds']/np.sqrt(numObs)
    summaryDF['CI_low'] = summaryDF['means'] - summaryDF['marginOfError']
    summaryDF['CI_high'] = summaryDF['means'] + summaryDF['marginOfError']
    return(summaryDF)

######################################################################################################################################################################
######### By Item Functions
######################################################################################################################################################################
def facetUtilityByNItem(df):
    dfItems = df.copy()
    dlen = lambda x: len(x['targetDictionary'])
    dfItems['nTargets'] = dfItems.apply(dlen, axis=1)
    
    modelNames = [modName for modName in dfItems.columns if 'utility' in modName]

    utility = pd.DataFrame(index = modelNames)
    utilityME = pd.DataFrame(index = modelNames)
    utilityDF = {}
    minItems = min(dfItems['nTargets'])
    maxItems = max(dfItems['nTargets'])+1

    for itemSetSize in range(minItems, maxItems):
        df_item = dfItems.loc[dfItems['nTargets'] == itemSetSize]
        print("items: ", itemSetSize, 'data points: ', df_item.shape[0])

        df_util = getPercentFromOptimalUtilityDF(df_item)
        utilityDF[str(itemSetSize)] = df_util
        
        utilSummary = getUtilityDifferenceSummaryStatistics(df_util)
        utilLabel = 'util' + str(itemSetSize)
        utility[utilLabel] = utilSummary['means']
        utilityME[utilLabel] = utilSummary['marginOfError']

    return(utility.transpose(), utilityME.transpose(), utilityDF)


def facetAccuracyByNItem(df):
    dfItems = df.copy()
    dlen = lambda x: len(x['targetDictionary'])
    dfItems['nTargets'] = dfItems.apply(dlen, axis=1)
    
    modelNames = [modName for modName in dfItems.columns if 'utility' in modName]
    
    accDF = {}
    minItems = min(dfItems['nTargets'])
    maxItems = max(dfItems['nTargets'])+1

    for itemSetSize in range(minItems, maxItems):
        df_item = dfItems.loc[dfItems['nTargets'] == itemSetSize]
        df_acc = getProportionTargetReached(df_item)        
        accDF[str(itemSetSize)] = df_acc
    return(accDF)

def facetQuitByNItem(df):
    dfItems = df.copy()
    getPropSignalerQuits =  lambda df, colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0
    modelNames = [modName for modName in dfItems.columns if 'sChoice' in modName]

    quitProp = pd.DataFrame(columns = modelNames)
    quitPropME = pd.DataFrame(columns = modelNames)

    for itemSetSize in range(min(dfItems['nTargets']), max(dfItems['nTargets'])+1):
        df_item = dfItems.loc[dfItems['nTargets'] == itemSetSize]
        print("items: ", itemSetSize, 'data points: ', df_item.shape[0])
        sigQuitsItem = [getPropSignalerQuits(df_item, mod) for mod in modelNames]
        quitProp.loc[itemSetSize] = sigQuitsItem
        
        sigDoesntQuit = [1-q for q in sigQuitsItem]
        p1_p_n = [p*q/dfItems.shape[0] for p, q in zip(sigQuitsItem,sigDoesntQuit)]
        quitPropME.loc[itemSetSize] =  1.96*np.sqrt(p1_p_n)

    return(quitProp, quitPropME)


######################################################################################################################################################################
######### Visualization Functions
######################################################################################################################################################################

def plotStackedCommunication(df, save=False, filename='./communicationBreakdown.png', rColor = '#484D60', sColor = '#9FA5BD', fSize = (7,6)):
    fig = plt.figure(figsize =fSize)
    ax = fig.add_axes([0,0,1,1])
    width=.5
    
    modelNames = list(df.index)
    y_signalerAchievesGoal = df['receiver failure'].values
    y_receiverAchievesGoal = df['receiver'].values
    me_rec = df['marginOfErrorR']
    me_sig = df['marginOfErrorRF']
    
    
    p1 = plt.bar(modelNames, y_receiverAchievesGoal, width,  yerr = me_rec, label = 'Successful Communication', color = rColor,edgecolor='white')
    p2 = plt.bar(modelNames, y_signalerAchievesGoal, width, yerr = me_sig,label = 'Failed Communication', color = sColor, edgecolor='white',bottom=y_receiverAchievesGoal)
    
    ax.set_yticks([0,.5,1])
    ax.set_xticklabels(modelNames, rotation=90)
    ax.set_ylabel('Communication Attempted (Proportion)')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    if save:
        fig.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()



def plotMetricByItem(accDf, meDF, save = False, filename = './acc.png', yAxisLabel = 'Proportion successful communication', yTicks = [0,.1, .2, .3, .4, .5,.6, .7, .8, .9, 1], fSize= (8,5),minItems = 2, maxItems = 9):
    fig = plt.figure(figsize =fSize)
    ax = fig.add_axes([0,0,1,1])
    labelDict = {'RSA_S0R0': 'Naive RSA', 
                'RSA_S1R1': 'Pragmatic RSA', 
                 'IW_S1R0': 'Naive IW',
                 'IW_S1R1': 'Pragmatic IW', 
                 'JU':'Joint Utility',#JU_sAchievesGoal
                 'DIYSignaler': 'Signaler Does', 
                 'CentralControl': 'Optimal'}

    colorDict = {'RSA_S0R0': '#feb954', 
                'RSA_S1R1': '#fed554', 
                 'IW_S1R0': 'orange',
                 'IW_S1R1': '#b10d2f', 
                 'JU': '#95a84c',
                 'DIYSignaler': "#91afec", 
                 'CentralControl': '#555555',
                 'Other': 'red'} 

    modelLabels = accDf.columns
    x = [a for a in range(minItems, maxItems+1)]
    for modelLabel in modelLabels:
        modelKey = [lab for lab in labelDict.keys() if lab in modelLabel]
        if len(modelKey) > 0:
            lab = modelKey[0]
            plt.errorbar(x, accDf[modelLabel].values, yerr = meDF[modelLabel].values, color = colorDict[lab],label=labelDict[lab], linewidth=3)
        else:
            plt.errorbar(x, accDf[modelLabel].values, yerr = meDF[modelLabel].values, color = colorDict['Other'],label=modelLabel, linewidth=3)

    plt.xlabel('Number of Items', size = 24 )
    plt.ylabel(yAxisLabel, size=24)
    #plt.legend(loc='best')
    ax.set_yticks(yTicks)
    ax.tick_params(labelsize=24)
    plt.ylim((-15,110))
    if save:
        fig.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()

def plotMetricByItem2(accDf, meDF, save = False, filename = './acc.png', yAxisLabel = 'Proportion successful communication', yTicks = [0,.1, .2, .3, .4, .5,.6, .7, .8, .9, 1], fSize= (8,5),minItems = 2, maxItems = 9):
    fig = plt.figure(figsize =fSize)
    ax = fig.add_axes([0.1,0.1,0.9,0.9])
    labelDict = {
                 'JU_sAchievesGoal':'Receiver Cost Ratio 0',
                 'JU_sAchievesGoalC_R0.2':'Receiver Cost Ratio 0.2',
                 'JU_sAchievesGoalC_R0.4':'Receiver Cost Ratio 0.4',
                 'JU_sAchievesGoalC_R0.6':'Receiver Cost Ratio 0.6',
                 'JU_sAchievesGoalC_R0.8':'Receiver Cost Ratio 0.8',
                 'JU_sAchievesGoalC_R1':'Receiver Cost Ratio 1',
                 }

    colorDict = {
                 'JU_sAchievesGoalC_R1': 'orange',
                 'JU_sAchievesGoalC_R0.8': '#b10d2f', 
                 'JU_sAchievesGoalC_R0.6': '#95a84c',
                 'JU_sAchievesGoalC_R0.4': "#91afec", 
                 'JU_sAchievesGoalC_R0.2': '#555555',
                 'JU_sAchievesGoal': '#feb954',
                 } 

    modelLabels = accDf.columns
    x = [a for a in range(minItems, maxItems+1)]
    for modelLabel in modelLabels:
        modelKey = [lab for lab in labelDict.keys() if lab in modelLabel]
        if len(modelKey) > 0:
            lab = modelKey[0]
            plt.errorbar(x, accDf[modelLabel].values, yerr = meDF[modelLabel].values, color = colorDict[lab],label=labelDict[lab], linewidth=3)
        else:
            plt.errorbar(x, accDf[modelLabel].values, yerr = meDF[modelLabel].values, color = colorDict['Other'],label=modelLabel, linewidth=3)

    plt.xlabel('Number of Items', size = 24 )
    plt.ylabel(yAxisLabel, size=24)
    #plt.legend(loc='best')
    ax.set_yticks(yTicks)
    ax.tick_params(labelsize=24)
    #plt.ylim((0,1))
    if save:
        fig.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()

#Subplot creation
def plotStackedCommunicationByItem(ax, accuracyDict, modelToPlot, modelName, sCanQuit = False,
                                   showLegend = True, succCommColor1 = "#000650", succCommColor2 = "#001650", failCommColor = '#9FA5BD', doColor = '#eaecf1', quitColor = '#f2dedd'):
    width=0.7
    nItems = list(accuracyDict.keys())
    nItems.sort()
    
    sig = []
    #succComm = []
    failComm = []
    succComm1 = []
    succComm2 = []
    
    sig_ME = []
    #succComm_ME = []
    succComm_ME1 = []
    succComm_ME2 = []
    failComm_ME = []

    if sCanQuit:
        sQuits = []
        sQuits_ME = []
    
    for item in nItems:
        itemDF = accuracyDict[item]
        modelRow = itemDF.loc[modelToPlot]
        
        sig.append(modelRow['signaler'])
        sig_ME.append(modelRow['marginOfErrorS'])
        succComm1.append(modelRow['receiver_1word'])
        succComm_ME1.append(modelRow['marginOfErrorR1'])
        succComm2.append(modelRow['receiver_2word'])
        succComm_ME2.append(modelRow['marginOfErrorR2'])
        failComm.append(modelRow['receiver failure'])
        failComm_ME.append(modelRow['marginOfErrorRF'])

        if sCanQuit:
            sQuits.append(modelRow['quit'])
            sQuits_ME.append(modelRow['marginOfErrorQ'])
    
    
    ax.bar(nItems, succComm1, width, label = 'Successful \nCommunication \n1 word', color = succCommColor1,edgecolor='white')
    ax.bar(nItems, succComm2, width, label = 'Successful \nCommunication \n2 words', color = succCommColor2,edgecolor='white', bottom = succComm1)
    g = [succ1 + succ2 for succ1, succ2 in zip(succComm1,succComm2)]#
    ax.bar(nItems, failComm, width, label = 'Failed \nCommunication', color = failCommColor, edgecolor='white', bottom=g)#
    
    heightToPlot = [succ1 + succ2 + fail for succ1, succ2, fail in zip(succComm1,succComm2, failComm)]
    ax.bar(nItems, sig, width, label = 'Signaler Does', color = doColor, edgecolor='white', bottom=heightToPlot)

    if sCanQuit:
        allExceptSQuitHeight = [sigDoes+cumulativeHeight for sigDoes, cumulativeHeight in zip(sig, heightToPlot)]
        ax.bar(nItems, sQuits, width, label = 'Signaler Quits', color = quitColor, edgecolor='white', bottom=allExceptSQuitHeight)
    
    ax.set_yticks([0,.5,1])
    ax.set_ylabel(modelName)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if showLegend:
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
