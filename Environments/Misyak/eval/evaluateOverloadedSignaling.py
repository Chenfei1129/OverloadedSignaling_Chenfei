import sys
sys.path.append('../src/')
sys.path.append('../envs/misyak/')

import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

from GenerativeSignaler import SignalerZero
from consistentSignalChecks_Misyak import signalIsConsistent_Boxes
from OverloadedReceiver import ReceiverZero
from OverloadedSignaler import SignalerOne

import namingConstants as NC
from misyakConstruction import *
from mindConstruction import *


class EvaluateOverloadedSignaler(object):
    def __init__(self, conditionDictionary, targetResult, fixedParameters):
        self.conditionDictionary = conditionDictionary
        self.targetResult = targetResult
        self.fixedParameters= fixedParameters

    def __call__(self, alpha, signalTypePrior):
        #functions that do not change across conditions
        getActionUtility = ActionUtility(self.fixedParameters['locationCost'], self.fixedParameters['rewardValue'], self.fixedParameters['nonRewardCost'])
        getActionDistribution = ActionDistributionGivenWorldGoal(alpha, getActionUtility, False)
        getMindSoftmax = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
        
        oddOneOutProportions = {}
        #set up signaler and receiver layers for each condition
        for conditionName, conditionParams in self.conditionDictionary.items():
            getGenerativeSignaler = self.constructConditionSignaler0(conditionParams)
            getReceiver0 = self.constructConditionReceiver0(signalTypePrior, conditionParams, getMindSoftmax, getGenerativeSignaler)
            getSignalerOne = self.constructConditionSignaler1(alpha, conditionParams, getActionUtility, getReceiver0)
            
            signalDistribution = getSignalerOne(self.fixedParameters['observation'])
            
            #proportion of signalers who use the odd one out signal
            proportion = self.getOddOneOutProportion(signalDistribution)
            oddOneOutProportions[conditionName] = proportion[0]
            
            #human mse
            errorColumnLabel = conditionName + '_error'
            conditionError = proportion - self.targetResult[conditionName]
            oddOneOutProportions[errorColumnLabel] = conditionError[0]

            #proportion of signalers who place tokens on the reward
            rewardColumnLabel = conditionName + '_goto'
            goToProportion = self.getGoToProportion(signalDistribution)
            oddOneOutProportions[rewardColumnLabel] = goToProportion[0]

            rewardErrorColumnLabel = conditionName + '_goto_error'
            conditionRewardError = goToProportion - self.targetResult[rewardColumnLabel]
            oddOneOutProportions[rewardErrorColumnLabel] = conditionRewardError[0]

        
        oddOneOutProportions['total_rmse'] = self.rmse(oddOneOutProportions, self.targetResult)
        oddOneOutProportions['error_variance'] = self.varianceOfPredictionError(oddOneOutProportions, self.targetResult)
        print("condition alpha", alpha, "signal type prior", signalTypePrior)
        return(pd.Series(oddOneOutProportions))

    def constructConditionSignaler0(self, paramDict):
        signalSpace = list(paramDict[NC.SIGNALS])
        signaler0Function = SignalerZero(signalSpace, signalIsConsistent_Boxes)
        return(signaler0Function)

    def constructConditionReceiver0(self, signalPrior, paramDict, getMind, getGenerativeSignaler):
        worldSpace = paramDict[NC.WORLDS]
        desireSpace = paramDict[NC.DESIRES]
        goalSpace = paramDict[NC.INTENTIONS]
        actionSpace = paramDict[NC.ACTIONS]
        
        commonGroundSpace = {NC.WORLDS: worldSpace, NC.DESIRES: desireSpace, NC.INTENTIONS: goalSpace, NC.ACTIONS: actionSpace}
        signalCategoryPrior = self.getSignalTypePriorDictionary(signalPrior)
        
        receiver0Function = ReceiverZero(commonGroundDictionary=commonGroundSpace, constructMind=getMind, getSignalerZero=getGenerativeSignaler, signalCategoryPrior=signalCategoryPrior)
        return(receiver0Function)
    
    def constructConditionSignaler1(self, alpha, paramDict, utilityFunction, receiver0Function):
        signalSpace = paramDict[NC.SIGNALS]
        signaler1Function = SignalerOne(alpha, signalSpace, utilityFunction, receiver0Function)
        return(signaler1Function)
        
    def getOddOneOutProportion(self, signalProbabilityDistribution):
        observation = self.fixedParameters['observation']
        trueWorld = observation[NC.WORLDS]
        oddOneOutSignal = tuple([1-x for x in trueWorld])
        return(signalProbabilityDistribution.loc[oddOneOutSignal].values[0])

    def getGoToProportion(self, signalProbabilityDistribution, nullSignal = (0,0,0)):
        observation = self.fixedParameters['observation']
        trueWorld = observation[NC.WORLDS]
        signals = list(signalProbabilityDistribution.index.get_level_values(NC.SIGNALS))

        isGotoSignal = lambda signal, world: all([w == 1 for u, w in zip(signal, world) if u == 1])
        goToSignals = [s for s in signals if isGotoSignal(s, trueWorld)]
        
        #remove null signal
        if nullSignal in goToSignals:
        	goToSignals.remove(nullSignal)

        probabilityOfGoToSignal = [signalProbabilityDistribution.loc[s].values[0] for s in goToSignals]
        return(sum(probabilityOfGoToSignal))

    def rmse(self, predictions, targets):
        squaredError = np.array([(predictions[condition] - targets[condition]) ** 2 for condition in targets.keys()])
        rootMeanSquaredError = np.sqrt((squaredError).mean())
        return(rootMeanSquaredError)
    
    def varianceOfPredictionError(self, predictions, targets):
        conditionErrors = np.array([(predictions[condition] - targets[condition]) for condition in targets.keys()])
        totalVariance = np.var(conditionErrors)
        return(totalVariance)
        
    def getSignalTypePriorDictionary(self, probability, categoryNames = ['1', '-1'], decimalPlaces = 7):
        prior = {categoryNames[0]:round(probability, decimalPlaces), categoryNames[1]: round(1-probability, decimalPlaces)}
        return(prior)


def drawPerformanceBarPlot(dfCondition, axisLabel, axForDraw, targetMeanDictionary, targetSEDictionary, conditionColors, nConditions = 4, hatches = None, barWidth = .35):
    N = nConditions
    ind = np.arange(N)  # the x locations for the groups
    width = barWidth      # the width of the bars
    #mpl.rcParams['hatch.color'] = 'grey'

    modelMeans = []
    targetMeans = []
    targetSEs = []
    cols = []
    hatchPattern = []

    for condition, trgtMean in targetMeanDictionary.items():
    	modelMeans.append(dfCondition.loc[axisLabel,condition])
    	targetMeans.append(trgtMean)
    	targetSEs.append(targetSEDictionary[condition])
    	cols.append(conditionColors[condition])
    	
    	if hatches is None:
    		hatchPattern.append('n')
    	else:
    		hatchPattern.append(hatches[condition])

    modelPreds = axForDraw.bar(ind, tuple(modelMeans), width, color=cols, alpha = .5)
    humanPreds = axForDraw.bar(ind + width, tuple(targetMeans), width, color=cols, yerr=tuple(targetSEs))

    for modelPred, humanPred, pattern in zip(modelPreds, humanPreds, hatchPattern):
        modelPred.set_hatch(pattern)
        humanPred.set_hatch(pattern)
    
    axForDraw.set_yticks([0,.5,1])
    axForDraw.set_xticks([])
    



if __name__ == "__main__":
	#Common ground and condition set up
	#world spaces
	twoRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 2)
	oneRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 1)
	wallWorldSpace = getWorldSpace(wall = True, nBoxes = 3, nRewards = 2)

	#action spaces
	oneAxActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 1)
	oneAxActionSpace.remove((0,0,0))
	twoAxActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 2)
	twoAxActionSpace.remove((0,0,0))

	#signal spaces
	oneTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 1)
	oneTokenSignalSpace.remove((0,0,0))
	twoTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 2)
	twoTokenSignalSpace.remove((0,0,0))

	#condition common ground spaces
	twoTokenCondition = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: twoAxActionSpace, NC.SIGNALS:twoTokenSignalSpace}
	inversionCondition = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: twoAxActionSpace, NC.SIGNALS:oneTokenSignalSpace}
	oneAxCondition = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: oneAxActionSpace, NC.SIGNALS:oneTokenSignalSpace}
	wallCondition = {NC.WORLDS: wallWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: twoAxActionSpace, NC.SIGNALS:oneTokenSignalSpace}

	conditionParameters = {'twoToken': twoTokenCondition,'inversion': inversionCondition,'wall': wallCondition,'oneAx': oneAxCondition}

	receiverHumanResults = {'inversion': .95, 'wall': .39, 'oneAx':.30, 'twoToken':.02}
	signalerHumanResults = {'inversion': .91, 'wall': .48, 'oneAx':.32, 'twoToken':.02}
	signalerHumanResultsExperiment2 = {'twoToken':0.0, 'twoToken_goto': 1.0, 'inversion':.56, 'inversion_goto':.42, 'wall':.23, 'wall_goto': .77, 'oneAx':.375, 'oneAx_goto': .625}

	fixedParameters = {'locationCost':0, 'rewardValue':5, 'nonRewardCost':0, 'observation': {NC.WORLDS:(1,1,0)}}

	#Evaluation of parameters in Misyak et al experiment 2
	#Tuning values - small grid
	manipulatedVariables = OrderedDict()
	manipulatedVariables['alpha'] = [6,8,10,12]
	manipulatedVariables['signalTypePriors'] = [.4, .5, .55, .6, .65]

	levelNames = list(manipulatedVariables.keys())
	levelValues = list(manipulatedVariables.values())
	modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
	toSplitFrame = pd.DataFrame(index=modelIndex)

	computeStatistics =  EvaluateOverloadedSignaler(conditionDictionary = conditionParameters, targetResult = signalerHumanResultsExperiment2, fixedParameters=fixedParameters)
	computeConditionStatistics = lambda condition: computeStatistics(condition.index.get_level_values('alpha')[0], condition.index.get_level_values('signalTypePriors')[0])
	conditionStatistics_4x5Expt2 = toSplitFrame.groupby(levelNames).apply(computeConditionStatistics)


	#plotting variables

	#experiment 1 target human performances from paper
	targetMeansExp1 = {'twoToken':0.02, 'inversion':.91, 'wall':.48, 'oneAx':.32}
	targetSEExp1 = {'twoToken':0.025, 'inversion':.042, 'wall':.073, 'oneAx':.082}

	#experiment 2 target human performances from paper
	targetMeansExp2 = {'twoToken':0.0, 'inversion':.56, 'wall':.23, 'oneAx':.38}
	targetSEExp2 = {'twoToken':0.0, 'inversion':.1, 'wall':.08, 'oneAx':.1}

	#plotting hyperparameters
	figureSize = (20,15)
	horizontalSpaceBetweenSubplots = .45
	legendColors = {'twoToken' : 'thistle','inversion':'royalblue','wall':'gold','oneAx':'lightsalmon'}

	#dataframe to plot
	dataframeName = conditionStatistics_4x5Expt2

	#overall figure drawing code
	fig = plt.figure(figsize=figureSize)
	numRows = len(manipulatedVariables['alpha'])
	numColumns = len(manipulatedVariables['signalTypePriors'])
	plotCounter = 1

	for alpha, grp in dataframeName.groupby('alpha'):
		grp.index = grp.index.droplevel('alpha')
		for signalTypePriors, group in grp.groupby('signalTypePriors'):
		
			#subplot axis labels
			axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
			if plotCounter % numColumns == 1:
				axForDraw.set_ylabel('alpha: {}'.format(alpha))
			if plotCounter <= numColumns:
				axForDraw.set_title('p(c) = 1: {}'.format(round(signalTypePriors, 2)))
			axForDraw.set_ylim(0, 1)

			#draw subplot
			drawPerformanceBarPlot(group, signalTypePriors, axForDraw, targetMeansExp2, targetSEExp2, legendColors)
			#drawPerformanceBarPlot(group, axForDraw, targetMeansExp2, targetSEExp2, legendColors)

			plotCounter += 1
	
	#Title
	plt.suptitle('Proportion Usage of Odd One Out Signaling')
	
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