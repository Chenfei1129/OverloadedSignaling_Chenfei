import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd 
import warnings

from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace
import Algorithms.constantNames as NC

# Bayesian inference for pragmatic receiver
class ReceiverOne(object):
    def __init__(self, commonGroundDictionary, constructMind, getSignalerZero, signalIsConsistent):
        self.mindPrior = constructMind(commonGroundDictionary)
        self.getSignalLikelihood = getSignalerZero
        self.signalIsConsistent = signalIsConsistent
        self.mindLabels = list(commonGroundDictionary.keys())
        

    def __call__(self, signal):
        posteriorDF = self.constructLikelihoodDataFrameFromMindConditions(self.mindPrior, signal)
        posteriorDF[NC.P_MINDPOSTERIOR] = posteriorDF[NC.P_MIND]*posteriorDF[NC.P_SIGNALONLYLIKELIHOOD]

        normalizingConstant = sum(posteriorDF[NC.P_MINDPOSTERIOR])
        #print(normalizingConstant)

        if round(normalizingConstant, 6) > 0: #temporary fix: if the normalizing constant is close enough to 0 assume there's no mass and dont normalize
            posteriorDF[NC.P_MINDPOSTERIOR] = posteriorDF[NC.P_MINDPOSTERIOR].apply(lambda x: x/normalizingConstant)
        else:
            warnings.warn('IW Prag Receiver - Normalizing a distribution w/o mass: irrational signal')

        return(posteriorDF[[NC.P_MINDPOSTERIOR]])

    
    def constructLikelihoodDataFrameFromMindConditions(self, mindPrior, signal):
        #for experiment intentions is uncertain -- this needs to be generalized
        uniqueIntentions = mindPrior.index.unique(level = NC.INTENTIONS)
        #print(uniqueIntentions)
        signalLikelihoods = {intention: self.getSignalLikelihood({NC.INTENTIONS: intention}).loc[signal].values[0][0] for intention in uniqueIntentions}
        #print(signalLikelihoods)
        #standard Bayesian inference - but also added a consistency notion?
        signalConsistent = lambda y: self.signalIsConsistent(signal, {NC.INTENTIONS: y.index.get_level_values(NC.INTENTIONS)[0], NC.ACTIONS:y.index.get_level_values(NC.ACTIONS)[0]})
        getConditionLikelihood = lambda x: signalLikelihoods[x.index.get_level_values(NC.INTENTIONS)[0]] #if signalConsistent(x) else 0.0    
        
        mindPrior[NC.P_SIGNALONLYLIKELIHOOD] = mindPrior.groupby(mindPrior.index.names).apply(getConditionLikelihood)
        #print(mindPrior)
        return(mindPrior)
    """
    def constructLikelihoodDataFrameFromMindConditions(self, mindPrior, signal): #for the misyak example
        #for experiment intentions is uncertain -- this needs to be generalized
        uniqueIntentions = mindPrior.index.unique(level = NC.WORLDS)
        print(uniqueIntentions)
        signalLikelihoods = {intention: self.getSignalLikelihood({NC.WORLDS: intention}).loc[signal].values[0][0] for intention in uniqueIntentions}
        #standard Bayesian inference - but also added a consistency notion?
        #signalConsistent = lambda y: self.signalIsConsistent(signal, {NC.INTENTIONS: y.index.get_level_values(NC.INTENTIONS)[0], NC.ACTIONS:y.index.get_level_values(NC.ACTIONS)[0]})
        getConditionLikelihood = lambda x: signalLikelihoods[x.index.get_level_values(NC.WORLDS)[0]] #if signalConsistent(x) else 0.0    
        
        mindPrior[NC.P_SIGNALONLYLIKELIHOOD] = mindPrior.groupby(mindPrior.index.names).apply(getConditionLikelihood)

        return(mindPrior)
    """
    
