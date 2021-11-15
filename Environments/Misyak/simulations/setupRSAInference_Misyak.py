import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import pandas as pd

import Algorithms.constantNames as NC

from Environments.Misyak.consistentSignalChecks_Misyak import signalIsConsistent_Boxes
from Environments.Misyak.misyakConstruction import *

import Algorithms.RSA.uncertainRSA as ursa


class SetupRSAMisyakTrial():
    def __init__(self, rationalityParameter, valueOfReward, 
                        signalMeaningPrior = {'1':.5, '-1':.5}, 
                        silencePossible=True, 
                        signalCost = 0, 
                        costOfPunishment = 0, 
                        nBoxes = 3):

        self.alpha = rationalityParameter
        self.valueOfReward = valueOfReward
        self.costOfPunishment = costOfPunishment
        
        self.nBoxes=nBoxes
        self.silencePossible = silencePossible
        self.signalCost= signalCost
        self.signalMeaningPrior = signalMeaningPrior
        
    def __call__(self, nTokens, nAxes, showShadow, nRewards):
        signalSpace, getSignalCost, statePrior = self.setupInferenceInputs(nTokens, nAxes, showShadow, nRewards)
        
        
        getS0 = ursa.LiteralSpeaker(lexicon=ursa.consistencyLexiconRSA_Misyak, 
                            messageSet=signalSpace, 
                            messageCostFunction = getSignalCost, 
                            lambdaRationalityParameter = self.alpha)
        
        getL1 = ursa.PragmaticListener(getSpeaker = getS0, 
                                       targetPrior = statePrior, 
                                       meaningPrior = self.signalMeaningPrior)

        getS1 = ursa.PragmaticSpeaker(getListener=getL1, 
                                      messageSet = signalSpace, 
                                      messageCostFunction= getSignalCost, 
                                      lambdaRationalityParameter = self.alpha)
        return(getS0, getL1, getS1)

    
    def setupInferenceInputs(self,nTokens, nAxes, showShadow, nRewards):
        signalSpace = self.buildSignalSpace(nTokens)
        getSignalCost = SignalCost_Misyak(self.signalCost) 
        unifStatePrior = self.buildWorldSpacePrior(nRewards, showShadow)
        return(signalSpace, getSignalCost, unifStatePrior)
        
    def buildSignalSpace(self, nTokens):
        signalSpace = getSignalSpace(nBoxes=self.nBoxes, nSignals = nTokens)
        if not self.silencePossible:
            signalSpace.remove((0,0,0))
        return(signalSpace)
    
    def buildWorldSpacePrior(self, nRewards, showShadow):
        wallPresent = bool((showShadow-1)*-1)
        worldSpace = getWorldSpace(wall = wallPresent, nBoxes = self.nBoxes, nRewards = nRewards)
        worldPrior = {world: 1/len(worldSpace) for world in worldSpace}
        return(worldPrior)


class SetupRSAMisyakTrial_Receiver():
    def __init__(self, rationalityParameter, valueOfReward, 
                        signalMeaningPrior = {'1':.5, '-1':.5}, 
                        silencePossible=True, 
                        signalCost = 0, 
                        costOfPunishment = 0, 
                        nBoxes = 3):
    
        self.alpha = rationalityParameter
        self.valueOfReward = valueOfReward
        self.costOfPunishment = costOfPunishment
        
        self.nBoxes=nBoxes
        self.silencePossible = silencePossible
        self.signalCost= signalCost
        self.signalMeaningPrior = signalMeaningPrior
        
    def __call__(self, nTokens, nAxes, showShadow, nRewards):
        signalSpace, getSignalCost, statePrior = self.setupInferenceInputs(nTokens, nAxes, showShadow, nRewards)
        
        getl0 = ursa.LiteralListener(worldPriorDictionary=statePrior, 
                                 meaningPriorDictionary=self.signalMeaningPrior, 
                                 lexiconFunction=ursa.consistencyLexiconRSA_Misyak)

        gets1 = ursa.PragmaticSpeaker_ListenerModel(getListener = getl0,
                                                    lexicon = ursa.consistencyLexiconRSA_Misyak, 
                                                    messageSet = signalSpace, 
                                                    messageCostFunction = getSignalCost, 
                                                    lambdaRationalityParameter = self.alpha)

        getl1 = ursa.PragmaticListener(getSpeaker = gets1, 
                                       targetPrior = statePrior, 
                                       meaningPrior = self.signalMeaningPrior)
        return(getl0, gets1, getl1)

    
    def setupInferenceInputs(self,nTokens, nAxes, showShadow, nRewards):
        signalSpace = self.buildSignalSpace(nTokens)
        getSignalCost = SignalCost_Misyak(self.signalCost) 
        unifStatePrior = self.buildWorldSpacePrior(nRewards, showShadow)
        return(signalSpace, getSignalCost, unifStatePrior)
        
    def buildSignalSpace(self, nTokens):
        signalSpace = getSignalSpace(nBoxes=self.nBoxes, nSignals = nTokens)
        if not self.silencePossible:
            signalSpace.remove((0,0,0))
        return(signalSpace)
    
    def buildWorldSpacePrior(self, nRewards, showShadow):
        wallPresent = bool((showShadow-1)*-1)
        worldSpace = getWorldSpace(wall = wallPresent, nBoxes = self.nBoxes, nRewards = nRewards)
        worldPrior = {world: 1/len(worldSpace) for world in worldSpace}
        return(worldPrior)