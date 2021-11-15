import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))

import numpy as np
import pandas as pd
import itertools as it
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

import Algorithms.constantNames as NC

from Environments.Misyak.simulations.setupInference_Misyak import SetupMisyakTrial
from Environments.Misyak.simulations.setupRSAInference_Misyak import SetupRSAMisyakTrial
from Environments.Misyak.consistentSignalChecks_Misyak import signalIsConsistent_Boxes
from Environments.Misyak.misyakConstruction import getSignalSpace


class EvaluateOverloadedSignaler(object):
    def __init__(self, humanResultsDataFrame, returnStrategyBreakdown=False, model = 'IW'):
        self.returnStrategyBreakdown = returnStrategyBreakdown
        self.humanResultsDataFrame = humanResultsDataFrame
        self.inferenceModel = model

    def __call__(self, alpha, valueOfReward, signalMeaningPrior, costOfPunishment=0, costOfSignal=0):

        if self.inferenceModel == 'IW':
            setupTrial = SetupMisyakTrial(rationalityParameter=alpha, 
                                        valueOfReward=valueOfReward, 
                                        signalMeaningPrior=self.formatSignalPrior(signalMeaningPrior),
                                        silencePossible=True, 
                                        signalCost = costOfSignal, 
                                        costOfPunishment = costOfPunishment,
                                        nBoxes = 3)
        else: #RSA model
            setupTrial = SetupRSAMisyakTrial(rationalityParameter=alpha, 
                                            valueOfReward=valueOfReward, 
                                            signalMeaningPrior = self.formatSignalPrior(signalMeaningPrior), 
                                            silencePossible=True, 
                                            signalCost = costOfSignal, 
                                            costOfPunishment = costOfPunishment,
                                            nBoxes = 3)

        results = self.humanResultsDataFrame.copy()
        results['modelResults'] = 0

        misyakExperimentConditions = list(it.product([1,2], [1,2], [0,1], [1,2])) #tokens, axes, show shadow, rewards - 16 experimental conditions
        for condition in misyakExperimentConditions:
            results.loc[condition,'modelResults'] =  self.getStrategyDistribution(condition, setupTrial)

        if self.returnStrategyBreakdown:
            return(results)
        else:
            conditionError, conditionVar = self.getRMSEandVariance(results, True)
            totalError, totalVar = self.getRMSEandVariance(results)
            misyakMetrics = {'keyConditionRMSE': conditionError, 'keyConditionVariance': conditionVar, 'totalRMSE': totalError, 'totalVariance': totalVar}
            print("alpha", alpha, "reward value", valueOfReward, "meaning prior", signalMeaningPrior, 'scorpion cost ', costOfPunishment, 'signal cost', costOfSignal)
            return(pd.Series(misyakMetrics))
        
    def getRMSEandVariance(self, resultsDF, conditionSubset = False):
        if conditionSubset:
            keyConditions = {'twoToken': self.createConditionFilter(resultsDF, (2,2,1,2)),
            'inversion': self.createConditionFilter(resultsDF, (1,2,1,2)),
            'axe':self.createConditionFilter(resultsDF, (1,1,1,2)),
            'wall':self.createConditionFilter(resultsDF, (1,2,0,2))}
            df = resultsDF.iloc[(keyConditions['twoToken'][0] & keyConditions['twoToken'][1] & keyConditions['twoToken'][2] & keyConditions['twoToken'][3]) | 
            (keyConditions['inversion'][0] & keyConditions['inversion'][1] & keyConditions['inversion'][2] & keyConditions['inversion'][3]) |
            (keyConditions['axe'][0] & keyConditions['axe'][1] & keyConditions['axe'][2] & keyConditions['axe'][3]) |
            (keyConditions['wall'][0] & keyConditions['wall'][1] & keyConditions['wall'][2] & keyConditions['wall'][3]) ]
        else:
            df = resultsDF.copy()

        error = df['humanResults'] - df['modelResults']
        rmse = np.sqrt(sum((error)**2)/df.shape[0])
        totalVariance = np.var(error)
        return(rmse, totalVariance)

    def getStrategyDistribution(self, conditionTuple, setupTrialFunction):
        signalDistribution = self.getSignalPDF(conditionTuple, setupTrialFunction)
        pAvoid = self.getProbabilityAvoidStrategy(signalDistribution, conditionTuple)
        pMixed = self.getProbabilityMixedStrategy(signalDistribution, conditionTuple)
        pOpen = self.getProbabilityOpenStrategy(signalDistribution, conditionTuple)
        pSilent = self.getProbabilitySilenceStrategy(signalDistribution)
        return([pAvoid, pMixed, pOpen, pSilent])
        #Returns array PDF of avoid, mixed, open, silent for strategy PDF given condition
              
    def getSignalPDF(self, conditionTuple, setupTrialFunction):
        ntokens = conditionTuple[0]
        axes = conditionTuple[1]
        showShadow = conditionTuple[2]
        nRewards = conditionTuple[3]
        #print("tokens", ntokens, "axes", axes, "showShadow", showShadow, 'nRewards', nRewards)
        _,_, signaler = setupTrialFunction(ntokens, axes, showShadow, nRewards)

        if self.inferenceModel == 'IW':
            if nRewards == 1:
                observedWorld = {NC.WORLDS:(1,0,0)}
            if nRewards == 2:
                observedWorld = {NC.WORLDS:(1,1,0)}
        else:
            if nRewards == 1:
                observedWorld = (1,0,0)
            if nRewards == 2:
                observedWorld = (1,1,0)

        signalPDF = signaler(observedWorld)
        return(signalPDF)
        
    def getProbabilityAvoidStrategy(self, signalProbabilityDistribution, conditionTuple, nullSignal=(0,0,0)):
        world = (1,0,0) if (conditionTuple[3] ==1) else (1,1,0)
        signalSet =  getSignalSpace(nBoxes=3, nSignals = conditionTuple[0])
        signalIsAvoid = lambda signal: all([box==0 for token, box in zip(signal, world) if token == 1])
        avoidSignals = [ s for s in signalSet if signalIsAvoid(s)]
        if nullSignal in avoidSignals:
            avoidSignals.remove(nullSignal)

        if self.inferenceModel == 'IW':
            probabilityOfAvoidSignal = [signalProbabilityDistribution.loc[s].values[0] for s in avoidSignals]
        else:
            probabilityOfAvoidSignal = [signalProbabilityDistribution.loc[[s]].values[0] for s in avoidSignals]

        return(sum(probabilityOfAvoidSignal)[0])

    def getProbabilityOpenStrategy(self, signalProbabilityDistribution, conditionTuple, nullSignal = (0,0,0)):
        world = (1,0,0) if (conditionTuple[3] ==1) else (1,1,0)
        signalSet =  getSignalSpace(nBoxes=3, nSignals = conditionTuple[0])
        signalIsOpen = lambda signal: all([box==1 for token, box in zip(signal, world) if token == 1])
        openSignals = [s for s in signalSet if signalIsOpen(s)]
        if nullSignal in openSignals:
            openSignals.remove(nullSignal)

        if self.inferenceModel == 'IW':
            probabilityOfOpenSignal = [signalProbabilityDistribution.loc[s].values[0] for s in openSignals]
        else:
            probabilityOfOpenSignal = [signalProbabilityDistribution.loc[[s]].values[0] for s in openSignals]
            
        return(sum(probabilityOfOpenSignal)[0])

    def getProbabilityMixedStrategy(self, signalProbabilityDistribution, conditionTuple, nullSignal = (0,0,0)):
        world = (1,0,0) if (conditionTuple[3] ==1) else (1,1,0)
        signalSet =  getSignalSpace(nBoxes=3, nSignals = conditionTuple[0])

        signalIsOpen = lambda signal: all([box==1 for token, box in zip(signal, world) if token == 1])
        openSignals = [s for s in signalSet if signalIsOpen(s)]
        signalIsAvoid = lambda signal: all([box==0 for token, box in zip(signal, world) if token == 1])
        avoidSignals = [ s for s in signalSet if signalIsAvoid(s)]

        allStrategySignals = openSignals + avoidSignals + [nullSignal]
        [signalSet.remove(sig) for sig in allStrategySignals if sig in signalSet]

        if len(signalSet) == 0:
            return(0.0)
        else:
            if self.inferenceModel == 'IW':
                probabilityOfMixedSignal = [signalProbabilityDistribution.loc[s].values[0] for s in signalSet]
            else:
                probabilityOfMixedSignal = [signalProbabilityDistribution.loc[[s]].values[0] for s in signalSet]

            return(sum(probabilityOfMixedSignal)[0])

    def getProbabilitySilenceStrategy(self, signalPDF,  nullSignal=(0,0,0)):
        if self.inferenceModel == 'IW':
            return(signalPDF.loc[nullSignal].values[0][0])
        else:
            return(signalPDF.loc[[nullSignal]].values[0][0])

    def createConditionFilter(self, df, condition):
        filter1 = df.index.get_level_values('n_tokens') == condition[0]
        filter2 = df.index.get_level_values('n_axes') == condition[1]
        filter3 =  df.index.get_level_values('show_imprints') == condition[2]
        filter4 = df.index.get_level_values('n_bananas') == condition[3]
        return([filter1, filter2, filter3, filter4])

    def formatSignalPrior(self, probability, categoryNames = ['1', '-1'], decimalPlaces = 7):
        prior = {categoryNames[0]:round(probability, decimalPlaces), categoryNames[1]: round(1-probability, decimalPlaces)}
        return(prior)
