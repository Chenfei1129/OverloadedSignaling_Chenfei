import pandas as pd
import numpy as np
import math


def findNumberofWordsinBaseline(row):
    #create list which is combo of all the different n length word feature combinations. Go through the target dictionary, for each remove the features that is not relevant to the true target.
    #new list of remaining signal that would be unique. Length of the shortest. 
    k = [m.split() for m in row['targetDictionary'].values()]
    tgt = row['intention']
    allFeatures = tgt.split()
    #print(allFeatures)
    #print(k)
    count_1 = 0
    count = []
    for j in range(len(allFeatures)):
        for i in range(len(k)):
            if allFeatures[j] in k[i]:
                count_1 = count_1 + 1
        count.append(count_1)
    if 1 in count:
        return 1
    count = []
    count_2 = 0

    for i in range(len(k)):
        if (allFeatures[0] in k[i] and allFeatures[1] in k[i]) :
            count_2 = count_2 + 1
    count.append(count_2)

    count_2 = 0
    
    for i in range(len(k)):
        if (allFeatures[0] in k[i] and allFeatures[2] in k[i]) :

            count_2 = count_2 + 1
    count.append(count_2)


    count_2 = 0

    for i in range(len(k)):
        if (allFeatures[1] in k[i] and allFeatures[2] in k[i]) :
            count_2 = count_2 + 1
    count.append(count_2)
    
    if 1 in count:
        return 2
###
    count = []
    count_3 = 0
    
    for i in range(len(k)):
        if (allFeatures[0] in k[i] and allFeatures[1] in k[i] and allFeatures[2] in k[i]) :
            count_3 = count_3 + 1
    count.append(count_3)
    
    count_3 = 0
    for i in range(len(k)):
        if (allFeatures[0] in k[i] and allFeatures[1] in k[i] and allFeatures[3] in k[i]) :
            count_3 = count_3 + 1
            
    count_3 = 0
    for i in range(len(k)):
        if (allFeatures[0] in k[i] and allFeatures[2] in k[i] and allFeatures[3] in k[i]) :
            count_3 = count_3 + 1
    count_3 = 0
    for i in range(len(k)):
        if (allFeatures[1] in k[i] and allFeatures[2] in k[i] and allFeatures[3] in k[i]) :
            count_3 = count_3 + 1
    if 1 in count:
        return 3

    return 4

    return(baselineWordLength)

def getBaselineActionChoice(dfrow):
    
        pass

"""
        signalCostString: a string that goes after IW_S1R0_sChoice to define the column to get signal length of
"""
def getSignalLength(dfrow, signalCostString, modelPrefix='IW_S1R0_sChoice'):
    colName = modelPrefix + signalCostString
    if 'do' in dfrow[colName]:#[0:2]:
        signalLength = 'do'
    elif 'quit' in dfrow[colName]:
        signalLength = 'quit'
    else:
        signalLength = len(dfrow[colName].split())
    return(signalLength)

def calculateBits(dfrow, columnName, numDimensions, numFeatures):
    #separate analysis of the data. summary. Raw number as a table. 
    if dfrow[columnName] == 'do' or dfrow[columnName] == 'quit':
        return 0
    product_length = 1
    for length in range(numDimensions + 1 - dfrow[columnName], numDimensions + 1):
        product_length = product_length * length
    bitsOfSignal = np.log2(product_length * (numFeatures ** dfrow[columnName]))
    return bitsOfSignal

def calculateBits2(dfrow, columnName, numDimensions, numFeatures):
    #separate analysis of the data. summary. Raw number as a table. 
    if dfrow[columnName] == 'do' or dfrow[columnName] == 'quit':
        return 0
    product_length = 1
    bitsOfSignal = dfrow[columnName] * 6
    return bitsOfSignal

def isDIYBetterThanCCWithCosts(dfrow, signalCost, baseline_columnName, cc_columnName = 'CentralControl_utility', diy_columnName='DIYSignaler_utility'):
        diyBetterThanCC =  (dfrow[cc_columnName] - dfrow[baseline_columnName] * signalCost < dfrow[diy_columnName])
        return(diyBetterThanCC)
    
def baselineChoice(dfrow, signalCost, baseline_columnName, cc_columnName = 'CentralControl_utility', diy_columnName='DIYSignaler_utility'):
        ccWithCost = dfrow[cc_columnName] - dfrow[baseline_columnName] * signalCost
        diyBetterThanCC =  dfrow[cc_columnName] - dfrow[baseline_columnName] * signalCost - dfrow[diy_columnName]
        baseChoice = 'Communicate'
        if (ccWithCost <= 0 and  dfrow[diy_columnName]<= 0):
            return 'quit'
        if (diyBetterThanCC <= 0):
            return 'DIY'
        return('Communicate')


def getTypeOfReceiverFailure(dfrow, communicate_columnName = 'IW_S1R0_sChoice', dIYBetterThanCCWithCosts = 'DIY_better_0', reach_target_columnName = 'IW_S1R0_goalAchieved'):
        if 'quit' in dfrow[communicate_columnName]:
                receiverFailureType = 'quit'
        elif 'do' in dfrow[communicate_columnName] and dfrow[dIYBetterThanCCWithCosts]:
                receiverFailureType = 'do_DIY_Optimal'
        elif 'do' in dfrow[communicate_columnName] and dfrow[dIYBetterThanCCWithCosts] == False:
                receiverFailureType = 'do_DIY_suboptimal'
        elif dfrow[reach_target_columnName] == False:
                receiverFailureType = 'communication_failure'
        else:
                receiverFailureType = False
        return(receiverFailureType)

def getUtilityWithSignalCost(dfrow, signalCost, communicate_columnName, cc_columnName, diy_columnName, reach_target_columnName):
        if 'quit' == dfrow[communicate_columnName]:
                return 0
        if 'do' in dfrow[communicate_columnName] and dfrow[reach_target_columnName]:
                return dfrow[diy_columnName]
        if 'do' in dfrow[communicate_columnName] and dfrow[reach_target_columnName] == False:
                return 0
        utility =  -len(list((dfrow[communicate_columnName].split()))) * signalCost + dfrow[cc_columnName]
        return(utility)

def getUtilityWithSignalCost_base(dfrow, signalCost, communicate_columnName, cc_columnName, diy_columnName, reach_target_columnName, numBase):
        if 'quit' == dfrow[communicate_columnName]:
                return 0
        if 'do' in dfrow[communicate_columnName]:
                return dfrow[diy_columnName]
        utility =  -numBase * signalCost + dfrow[cc_columnName]
        return(utility)
