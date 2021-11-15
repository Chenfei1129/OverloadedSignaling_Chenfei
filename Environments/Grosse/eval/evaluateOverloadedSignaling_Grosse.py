import sys
sys.path.append('../src/')

import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

from GenerativeSignaler import SignalerZero
from OverloadedReceiver import ReceiverZero
from OverloadedSignaler import SignalerOne

import namingConstants as NC
from consistentSignalChecks import signalIsConsistent_Grosse
from grosseConstruction import *
from mindConstruction import *


class EvaluateOverloadedSignaler(object):
    def __init__(self, conditionDictionary, targetResult, fixedParameters):
        self.conditionDictionary = conditionDictionary
        self.targetResult = targetResult
        self.fixedParameters = fixedParameters

    def __call__(self, alpha, goalProbLeftBattery, payoffScheme):
        #putility function
        utilityCosts, utilityRewards = self.transformPayoffSchemeToUtilities(payoffScheme)
        getUtility = ActionUtility_Grosse(utilityCosts, utilityRewards)
        
        #p(a|goal, world) function
        getActionPDF = ActionDistributionGivenWorldGoal_Grosse(alpha, getUtility, True)
        
        #p(goal) function
        goalPrior = self.getGoalDictionary(goalProbLeftBattery)
        getGoalProbability = GoalDistribution_NonUniform(goalPrior)
        
        #p(mind) combination
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalProbability, getActionPDF)
        
        distObjectGoalProbability = {}
        for conditionName, conditionParams in self.conditionDictionary.items():
            #inference steps
            getGenerativeSignaler = self.constructConditionSignaler0(conditionParams)
            getReceiver0 = self.constructConditionReceiver0(conditionParams, getMind, getGenerativeSignaler)
          
            mindDistribution = getReceiver0(self.fixedParameters['observedSignal'])
        
            #proportion of receivers who get the distant battery
            proportion = self.getDistantBatteryProportion(mindDistribution)
            distObjectGoalProbability[conditionName] = proportion
            
            #human mse
            errorColumnLabel = conditionName + '_error'
            conditionError = proportion - self.targetResult[conditionName]
            distObjectGoalProbability[errorColumnLabel] = conditionError
           
        distObjectGoalProbability['total_rmse'] = self.rmse(distObjectGoalProbability, self.targetResult)
        distObjectGoalProbability['error_variance'] = self.varianceOfPredictionError(distObjectGoalProbability, self.targetResult)
        print("condition alpha", alpha, "goal prior", goalProbLeftBattery)
        return(pd.Series(distObjectGoalProbability))
        
    def constructConditionSignaler0(self, paramDict):
        signalSpace = list(paramDict['signals'])
        signaler0Function = SignalerZero(signalSpace, signalIsConsistent_Grosse)
        return(signaler0Function)

    def constructConditionReceiver0(self, paramDict, getMind, getGenerativeSignaler):
        worldSpace = paramDict[NC.WORLDS]
        desireSpace = paramDict[NC.DESIRES]
        goalSpace = paramDict[NC.INTENTIONS]
        actionSpace = paramDict[NC.ACTIONS]
        
        commonGroundSpace = {NC.WORLDS: worldSpace, NC.DESIRES: desireSpace, NC.INTENTIONS: goalSpace, NC.ACTIONS: actionSpace}
        signalCategoryPrior = paramDict['signalTypePrior']
        
        receiver0Function = ReceiverZero(commonGroundSpace, getMind, getGenerativeSignaler, signalCategoryPrior)
        return(receiver0Function)
        
    def getDistantBatteryProportion(self, mindDistribution):
        actionDistribution = pd.DataFrame(mindDistribution.groupby(level=[NC.ACTIONS]).sum())
        indexLevels = list(actionDistribution.index.get_level_values(0))
        receiverGetsFarBatteryIndices = [indx for indx in indexLevels if indx[1]=='L']
        
        receiverGetsLeftBattery = actionDistribution.loc[receiverGetsFarBatteryIndices]
        probabilityReceiverGetsLeftBattery = receiverGetsLeftBattery.sum().values[0]
        return(probabilityReceiverGetsLeftBattery)

    def rmse(self, predictions, targets):
        squaredError = np.array([(predictions[condition] - targets[condition]) ** 2 for condition in targets.keys()])
        rootMeanSquaredError = np.sqrt((squaredError).mean())
        return(rootMeanSquaredError)
    
    def varianceOfPredictionError(self, predictions, targets):
        conditionErrors = np.array([(predictions[condition] - targets[condition]) for condition in targets.keys()])
        totalVariance = np.var(conditionErrors)
        return(totalVariance)

    def transformPayoffSchemeToUtilities(self, payoffScheme, actions=('L', 'R', 'n')):
        payoffTuples = payoffScheme[0]
        payoffDictionary = [{a : payoff for a, payoff in zip(actions, agentPayoff)} for agentPayoff in payoffTuples] 
        reward = payoffScheme[1]
        return([payoffDictionary, reward])
    
    def getGoalDictionary(self, probLeftBattery, categoryNames = ['L', 'R'], decimalPlaces = 7):
        prior = {categoryNames[0]:round(probLeftBattery, decimalPlaces), categoryNames[1]: round(1-probLeftBattery, decimalPlaces)}
        return(prior)

class GoalDistribution_NonUniform(object):
    def __init__(self, goalProbabilityDictionary):
        self.goalProbabilities = goalProbabilityDictionary

    def __call__(self, goalSpace, world, desire):
        goalDict = {NC.INTENTIONS: goalSpace}
        goalSpaceDF = getMultiIndexMindSpace(goalDict)
        getConditionGoalProbability = lambda x: self.goalProbabilities[x.index.get_level_values(NC.INTENTIONS)[0]]
        
        goalProbabilities = goalSpaceDF.groupby(goalSpaceDF.index.names).apply(getConditionGoalProbability)
        goalSpaceDF[NC.P_INTENTION] = goalSpaceDF.index.get_level_values(0).map(normalizeValuesPdSeries(goalProbabilities).get)
        return(goalSpaceDF)

def drawPerformanceBarPlot(dfCondition, axLabel, axForDraw, targetMeanDictionary, targetSEDictionary, conditionColors, nConditions = 4, barWidth = .35):
    N = nConditions
    ind = np.arange(N)  # the x locations for the groups
    width = barWidth      # the width of the bars

    modelMeans = []
    targetMeans = []
    targetSEs = []
    cols = []
    
    for condition, trgtMean in targetMeanDictionary.items():
        modelMeans.append(dfCondition.loc[axLabel, condition])
        targetMeans.append(trgtMean)
        targetSEs.append(targetSEDictionary[condition])
        cols.append(conditionColors[condition])

    modelPreds = axForDraw.bar(ind, tuple(modelMeans), width, color=cols, alpha = .5)
    humanPreds = axForDraw.bar(ind + width, tuple(targetMeans), width, color=cols, yerr=tuple(targetSEs))
    
    axForDraw.set_yticks([0,.5,1])
    axForDraw.set_xticks([])

if __name__ == "__main__":
    #condition common ground spaces
    handsFreeCondition = {NC.WORLDS: ['LR'], NC.DESIRES: [1], NC.INTENTIONS: ['L', 'R'], NC.ACTIONS: [('n', 'n'), ('L', 'n'), ('R', 'n'), ('n', 'L'), ('n', 'R'), ('L', 'R'), ('R', 'L')], 'signals': ['you', 'null'], 'signalTypePrior':{'1':1.0}}
    handsOccupiedCondition = {NC.WORLDS: ['LR'], NC.DESIRES: [1], NC.INTENTIONS: ['L', 'R'], NC.ACTIONS: [('n', 'n'), ('n', 'L'), ('n', 'R')], 'signals': ['you', 'null'], 'signalTypePrior':{'1':1.0}}

    conditionParameters = {'handsFree': handsFreeCondition,'handsOccupied': handsOccupiedCondition}
    receiverHumanResults = {'handsFree': .49,'handsOccupied': .23}

    fixedParameters = {'observation':{NC.WORLDS:'LR', NC.INTENTIONS:'L'}, 'observedSignal':'you'}

    #parameters for tuning
    manipulatedVariables = OrderedDict()
    manipulatedVariables['alpha'] = list(np.linspace(0.1,1.5,8))
    manipulatedVariables['goalPriors'] = list(np.linspace(.2, .4, 5))

    # payoff scheme: ((signalerLeft, signalerRight, signalerNone), (receiverLeft, receiverRight, receiverNone), reward)
    manipulatedVariables['payoffs'] = [(((-6,-1,0),(-3,-3,0)), 6), (((-5,-1,0),(-2,-2,0)), 6)]

    # evaluation
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    computeStatistics =  EvaluateOverloadedSignaler(conditionDictionary = conditionParameters, targetResult = receiverHumanResults, fixedParameters=fixedParameters)
    computeConditionStatistics = lambda condition: computeStatistics(condition.index.get_level_values('alpha')[0], condition.index.get_level_values('goalPriors')[0],  condition.index.get_level_values('payoffs')[0])
    conditionStatistics_Grosse = toSplitFrame.groupby(levelNames).apply(computeConditionStatistics)

    # separate plotting for different utility payoff matrices
    grosse_payoff1 = conditionStatistics_Grosse.xs(manipulatedVariables['payoffs'][0],level = 'payoffs', drop_level=True)
    if len(manipulatedVariables['payoffs']) > 1:
        grosse_payoff2 = conditionStatistics_Grosse.xs(manipulatedVariables['payoffs'][1],level = 'payoffs', drop_level=True)

    #target toddler performances from paper - proportion of children who get the far battery with SE
    targetMeansExp1 = {'handsFree':.49, 'handsOccupied':.23}
    targetSEExp1 = {'handsFree':0.29, 'handsOccupied':.34}


    #plotting hyperparameters
    figureSize = (20,15)
    horizontalSpaceBetweenSubplots = .45
    legendColors = {'handsFree' : 'lightsalmon', 'handsOccupied':'royalblue'}
    numberOfConditions = len(list(legendColors.keys()))

    #dataframe to plot
    dataframeName = grosse_payoff1
    dataFrameRowLength = len(manipulatedVariables['alpha'])
    dataFrameColLength = len(manipulatedVariables['goalPriors'])

    fig = plt.figure(figsize=figureSize)
    numRows = dataFrameRowLength
    numColumns = dataFrameColLength
    plotCounter = 1

    for alpha, grp in dataframeName.groupby('alpha'):
        grp.index = grp.index.droplevel('alpha')
        for goalPriors, group in grp.groupby('goalPriors'):

            #subplot axis labels
            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel('alpha: {}'.format(round(alpha, 2)))
            if plotCounter <= numColumns:
                axForDraw.set_title('p(far): {}'.format(round(goalPriors, 2)))
            axForDraw.set_ylim(0, 1)

            #draw subplot
            drawPerformanceBarPlot(group, goalPriors, axForDraw, targetMeansExp1, targetSEExp1, legendColors, numberOfConditions)

            plotCounter += 1

    #Title
    plt.suptitle('Proportion Receivers Getting Far Battery')

    #legend
    legend_dict = legendColors
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)
    plt.legend(handles=patchList, loc='best')

    #adjust horizontal spacing of subplots
    fig.subplots_adjust(wspace=horizontalSpaceBetweenSubplots)

    plt.show()