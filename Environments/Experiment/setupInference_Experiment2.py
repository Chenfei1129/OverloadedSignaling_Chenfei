import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import *
from Algorithms.ImaginedWe.GenerativeSignaler import SignalerZero
from Algorithms.ImaginedWe.OverloadedReceiver import ReceiverZero
from Algorithms.ImaginedWe.OverloadedSignaler import SignalerOne
from Algorithms.ImaginedWe.PragmaticReceiver import ReceiverOne
from Environments.Experiment.consistentSignalChecks_Experiment2 import SignalIsConsistent_Experiment, SignalIsConsistent_ExperimentSeparatedSignals
from Environments.Experiment.experimentConstruction import *

##############################################################
########## Action space changed to move Signals to vocab only
##############################################################
########## General formulation with N Layers
################################################################

class SetupExperiment_SignalsSeparated_Levels():
    def __init__(self, beta, valueOfReward, getUtility = JointActionUtility, getCost = calculateLocationCost_TaxicabMetric, signalerInactionPossible = False, signalCost = None, receiverCostRatio = None):
        self.beta = beta
        self.valueOfReward = valueOfReward

        self.getUtility = getUtility
        self.getCost = getCost

        self.signalerInactionPossible = signalerInactionPossible
        self.signalCost = signalCost

        self.receiverCostRatio = receiverCostRatio


    def __call__(self, signalerLocation, receiverLocation, signalSpace, targetDictionary, maxLayer = 1, getAllLayers = True):
        getUtility, getMind, getSignalConsistency, mindCondition, signalMeaningPrior = self.combineInferenceInputs(signalerLocation, receiverLocation, targetDictionary)
        currentLayer = 0

        getSignaler = SignalerZero(
            signalSpace = signalSpace, 
            signalIsConsistent = getSignalConsistency, 
            signalCostFunction = self.signalCost)

        getReceiver = ReceiverZero(
            commonGroundDictionary=mindCondition, 
            constructMind=getMind, 
            getSignalerZero=getSignaler, 
            signalCategoryPrior=signalMeaningPrior)

        #getListener = LiteralListener(worldPriorDictionary=targetPrior, lexiconFunction=self.lexicon)
        listenerDict = {str(currentLayer): getReceiver}
        signalerDict = {str(currentLayer): getSignaler}

        while currentLayer < maxLayer:
            getSignaler = SignalerOne(alpha=self.beta, 
                signalSpace=signalSpace, 
                getActionUtility=getUtility, 
                getReceiverZero=getReceiver, 
                getSignalCost=self.signalCost)

            getSignaler_ActionsIncluded = AddDoYourselfSignalerUtility(getSignalerUtility = getSignaler,
                                                                    alpha=self.beta, 
                                                                    rewardValue=self.valueOfReward, 
                                                                    signalerPosition = signalerLocation, 
                                                                    targetDictionary=targetDictionary,
                                                                    costFunction=self.getCost,
                                                                    signalerInactionPossible = self.signalerInactionPossible)

            getReceiver = ReceiverOne(commonGroundDictionary=mindCondition, 
                constructMind = getMind, 
                getSignalerZero = getSignaler, 
                signalIsConsistent = getSignalConsistency)

            #getPragmaticSpeaker = PragmaticSpeaker(getListener=getListener, messageSet=signalSpace, messageCostFunction=self.getCost, lambdaRationalityParameter=self.rationality)
            #getListener  = PragmaticListener(getSpeaker = getPragmaticSpeaker, targetPrior = targetPrior)
            currentLayer += 1
            listenerDict[str(currentLayer)] = getReceiver
            signalerDict[str(currentLayer)] = getSignaler_ActionsIncluded
        if getAllLayers:
            return(signalerDict, listenerDict)
        else:
            return(signalerDict[str(currentLayer)],listenerDict[str(currentLayer)])


    def combineInferenceInputs(self, locS, locR, targetDictionary):
        getUtility = self.setupUtility(locS, locR, targetDictionary)    
        getMind = self.setupMind(getUtility)
        conditionDict = self.setupConditionDictionary(locS, locR, targetDictionary)
        signalCategoryPrior = {'1':1}
        signalIsConsistent = SignalIsConsistent_ExperimentSeparatedSignals(targetDictionary, locS, locR)
        return(getUtility, getMind, signalIsConsistent, conditionDict, signalCategoryPrior)


    def setupUtility(self, locS, locR, targetDictionary):
        utilityFunction = self.getUtility(
            costFunction = self.getCost, 
            valueOfReward = self.valueOfReward, 
            signalerLocation=locS, 
            receiverLocation=locR, 
            targetDictionary= targetDictionary)
        return(utilityFunction)

    def setupMind(self, utilityFunction):
        getActionDistribution = ActionDistributionGivenWorldGoal(self.beta, utilityFunction)
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, 
            getDesireProbability_Uniform, 
            getGoalGivenWorldAndDesire_Uniform, 
            getActionDistribution)
        return(getMind)

    def setupConditionDictionary(self, signalerLocation, receiverLocation, targetDictionary):
        actionSpace = getActionSpace_SignalsSeparated(targetDictionary=targetDictionary,
            signalerPosition = signalerLocation, 
            receiverPosition = receiverLocation)
        intentionSpace = list(targetDictionary.values())
        conditionDict = {NC.WORLDS: [1], NC.DESIRES: [1], NC.INTENTIONS: intentionSpace, NC.ACTIONS: actionSpace}
        return(conditionDict)


class SetupExperiment_SignalsSeparated():
    def __init__(self, beta, valueOfReward, getUtility = JointActionUtility, getCost = calculateLocationCost_TaxicabMetric, signalerInactionPossible = False, signalCost = None, receiverCostRatio = None):
        self.beta = beta
        self.valueOfReward = valueOfReward

        self.getUtility = getUtility
        self.getCost = getCost
        
        self.signalerInactionPossible = signalerInactionPossible
        self.signalCost = signalCost

        self.receiverCostRatio = receiverCostRatio
        #self.getCost = CalculateLocationCost_TaxicabMetric_receiverCostRatio(self.receiverCostRatio)
        #print(self.receiverCostRatio)



    def __call__(self, signalerLocation, receiverLocation, signalSpace, targetDictionary, genSig = False):
        getUtility, getMind, getSignalConsistency, mindCondition, signalMeaningPrior = self.combineInferenceInputs(signalerLocation, receiverLocation, targetDictionary)

        getGenerativeSignaler = SignalerZero(
            signalSpace = signalSpace, 
            signalIsConsistent = getSignalConsistency, 
            signalCostFunction = self.signalCost)

        getReceiverZero = ReceiverZero(
            commonGroundDictionary=mindCondition, 
            constructMind=getMind, 
            getSignalerZero=getGenerativeSignaler, 
            signalCategoryPrior=signalMeaningPrior)

        getSignalerOne = SignalerOne(
            alpha=self.beta, 
            signalSpace=signalSpace, 
            getActionUtility=getUtility, 
            getReceiverZero=getReceiverZero, 
            getSignalCost=self.signalCost)

        getSignaler_ActionsIncluded = AddDoYourselfSignalerUtility(getSignalerUtility = getSignalerOne,
                                                                    alpha = self.beta, 
                                                                    rewardValue = self.valueOfReward, 
                                                                    signalerPosition = signalerLocation, 
                                                                    targetDictionary = targetDictionary,receiverCostRatio = self.receiverCostRatio,
                                                                    costFunction = self.getCost, 
                                                                    signalerInactionPossible = self.signalerInactionPossible)
        getReceiverOne = ReceiverOne(
            commonGroundDictionary=mindCondition, 
            constructMind=getMind, 
            getSignalerZero=getSignalerOne, 
            signalIsConsistent =getSignalConsistency)

        if genSig:
            return(getGenerativeSignaler, getReceiverZero, getSignaler_ActionsIncluded, getReceiverOne)
        else:
            return(getReceiverZero, getSignaler_ActionsIncluded, getReceiverOne)

    def combineInferenceInputs(self, locS, locR, targetDictionary):
        getUtility = self.setupUtility(locS, locR, targetDictionary)
        
        getMind = self.setupMind(getUtility)
        conditionDict = self.setupConditionDictionary(locS, locR, targetDictionary)
        signalCategoryPrior = {'1':1}
        signalIsConsistent = SignalIsConsistent_ExperimentSeparatedSignals(targetDictionary, locS, locR)
        return(getUtility, getMind, signalIsConsistent, conditionDict, signalCategoryPrior)


    def setupUtility(self, locS, locR, targetDictionary):
        if self.receiverCostRatio == None:
            utilityFunction = self.getUtility(
            costFunction = self.getCost, 
            valueOfReward = self.valueOfReward, 
            signalerLocation=locS, 
            receiverLocation=locR,  
            targetDictionary= targetDictionary)
        else:
            utilityFunction = self.getUtility(
            costFunction = self.getCost, 
            valueOfReward = self.valueOfReward, 
            signalerLocation=locS, 
            receiverLocation=locR,  
            targetDictionary= targetDictionary, receiverCostRatio = self.receiverCostRatio)
        return(utilityFunction)

    def setupMind(self, utilityFunction):
        getActionDistribution = ActionDistributionGivenWorldGoal(self.beta, utilityFunction)
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, 
            getDesireProbability_Uniform, 
            getGoalGivenWorldAndDesire_Uniform, 
            getActionDistribution)
        return(getMind)

    def setupConditionDictionary(self, signalerLocation, receiverLocation, targetDictionary):
        actionSpace = getActionSpace_SignalsSeparated(targetDictionary=targetDictionary,
            signalerPosition = signalerLocation, 
            receiverPosition = receiverLocation)
        intentionSpace = list(targetDictionary.values())
        conditionDict = {NC.WORLDS: [1], NC.DESIRES: [1], NC.INTENTIONS: intentionSpace, NC.ACTIONS: actionSpace}
        return(conditionDict)
    
    def calculateSignalCost(signal, mind, cost = 1):
        return (len(test_string.split()) *cost)
