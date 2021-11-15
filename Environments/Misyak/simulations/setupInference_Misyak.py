import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import pandas as pd

import Algorithms.constantNames as NC

from Environments.Misyak.consistentSignalChecks_Misyak import signalIsConsistent_Boxes
from Environments.Misyak.misyakConstruction import *

from Algorithms.ImaginedWe.mindConstruction import *
from Algorithms.ImaginedWe.GenerativeSignaler import SignalerZero
from Algorithms.ImaginedWe.OverloadedReceiver import ReceiverZero
from Algorithms.ImaginedWe.OverloadedSignaler import SignalerOne
from Algorithms.ImaginedWe.PragmaticReceiver import ReceiverOne

class SetupMisyakTrial():
    def __init__(self, rationalityParameter, valueOfReward, signalMeaningPrior = {'1':.5, '-1':.5}, silencePossible=True, signalCost = 0, costOfPunishment = 0,nBoxes = 3):
        self.alpha = rationalityParameter
        self.valueOfReward = valueOfReward
        self.costOfPunishment = costOfPunishment
        
        self.nBoxes=nBoxes
        self.silencePossible = silencePossible
        self.signalCost= signalCost
        self.signalMeaningPrior = signalMeaningPrior
        
    def __call__(self, nTokens, nAxes, showShadow, nRewards, recOne = False):
        signalSpace, condition, getUtility, getActionPDF, getMind, getSignalCost = self.setupInferenceInputs(nTokens, nAxes, showShadow, nRewards)
        
        getSignalerZero = SignalerZero(signalSpace = signalSpace, 
            signalIsConsistent = signalIsConsistent_Boxes, 
            signalCostFunction = getSignalCost)

        getReceiverZero = ReceiverZero(commonGroundDictionary=condition, 
            constructMind=getMind, 
            getSignalerZero=getSignalerZero, 
            signalCategoryPrior=self.signalMeaningPrior)

        getSignalerOne = SignalerOne(alpha=self.alpha, 
            signalSpace =signalSpace,  
            getActionUtility=getUtility, 
            getReceiverZero=getReceiverZero,
            signalIsConsistent= None,
            getSignalCost = getSignalCost)

        getReceiverOne = ReceiverOne(commonGroundDictionary=condition, 
            constructMind=getMind, 
            getSignalerZero=getSignalerOne, 
            signalIsConsistent=signalIsConsistent_Boxes)
        if recOne:
            return(getSignalerZero, getReceiverZero, getSignalerOne, getReceiverOne)
        else:
            return(getSignalerZero, getReceiverZero, getSignalerOne)

    
    def setupInferenceInputs(self,nTokens, nAxes, showShadow, nRewards):
        signalSpace = self.buildSignalSpace(nTokens)
        condition = self.buildConditionDictionary(nRewards,showShadow, nAxes)
        actionUtilityFunction = ActionUtility(costOfLocation=0, costOfNonReward=self.costOfPunishment, valueOfReward=self.valueOfReward)
        getActionDistribution = ActionDistributionGivenWorldGoal(self.alpha, actionUtilityFunction, False)
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
        getSignalCost = SignalCost_Misyak(self.signalCost) 
        
        return(signalSpace, condition, actionUtilityFunction, getActionDistribution, getMind, getSignalCost)
        
    def buildSignalSpace(self, nTokens):
        signalSpace = getSignalSpace(nBoxes=self.nBoxes, nSignals = nTokens)
        if not self.silencePossible:
            signalSpace.remove((0,0,0))
        return(signalSpace)
        
    def buildConditionDictionary(self, nRewards, showShadow, nAxes):
        worlds = self.buildWorldSpace(nRewards, showShadow)
        actions = self.buildActionSpace(nAxes)
        conditionDict = {NC.WORLDS: worlds, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: actions}
        return(conditionDict)
    
    def buildWorldSpace(self, nRewards, showShadow):
        wallPresent = bool((showShadow-1)*-1)
        worldSpace = getWorldSpace(wall = wallPresent, nBoxes = self.nBoxes, nRewards = nRewards)
        return(worldSpace)
    
    def buildActionSpace(self, nAxes):
        actionSpace = getActionSpace(nBoxes = self.nBoxes, nReceiverChoices = nAxes)
        if not self.silencePossible:
            actionSpace.remove((0,0,0))
        return(actionSpace)