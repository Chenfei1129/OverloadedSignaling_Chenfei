import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np 
import pandas as pd

import Algorithms.constantNames as NC

"""
    signalSpace: list of all possible signals, inpit type must be valid pandas index
    signalIsConsistent: function specifying whether the signal is logically consistent given a mind and signaler category
"""
class SignalerZero(object):
    def __init__(self, signalSpace, signalIsConsistent, signalCostFunction=None):
        #print(signalSpace)
        self.signalSpace = signalSpace
        self.signalIsConsistent = signalIsConsistent
        self.getSignalCost = signalCostFunction

    def __call__(self, targetMind, signalerCategory): #p(signal|mind, category), for all signals in signal space
        #create a dataframe that adds signal as an index to the target mind and a column labeled p(signal|mind,c)
        self.signalSpace.append('purple circle')
        likelihoodComponents = pd.DataFrame(data=np.inf,index=targetMind.index, columns=self.signalSpace).stack()
        likelihoodComponents.index.names = targetMind.index.names + [NC.SIGNALS]
        likelihoodComponents.name = NC.P_SIGNALLIKELIHOOD

        #for each condition apply the get likelihood function, returns a distribution
        signalLikelihoods = likelihoodComponents.groupby(likelihoodComponents.index.names).apply(self.getSignalLikelihoodGivenMind, signalerType = signalerCategory)
        signalLikelihoods = pd.DataFrame(signalLikelihoods)
        
        
        #levelsToNormalizeOver = [signalLikelihoods.index.get_level_values(NC.INTENTIONS), signalLikelihoods.index.get_level_values(NC.ACTIONS)]
        #signalLikelihoods[NC.P_SIGNALLIKELIHOOD] /= signalLikelihoods.groupby(levelsToNormalizeOver)[NC.P_SIGNALLIKELIHOOD].transform(sum)
        #print(signalLikelihoods.iloc[:,1])
        return(signalLikelihoods)

    def getSignalLikelihoodGivenMind(self, signalingCondition, signalerType):
        world = signalingCondition.index.get_level_values(NC.WORLDS)[0]
        desire = signalingCondition.index.get_level_values(NC.DESIRES)[0]
        goal = signalingCondition.index.get_level_values(NC.INTENTIONS)[0]
        action = signalingCondition.index.get_level_values(NC.ACTIONS)[0]
        mind = {NC.WORLDS: world, NC.DESIRES:desire, NC.INTENTIONS:goal, NC.ACTIONS:action}

        #extract the world and signal from the index condition
        signal = signalingCondition.index.get_level_values(NC.SIGNALS)[0]
        
        #check if signal is consistent with signaler type and mind, if so return 1/size of possible consisent signals
        #print(self.signalIsConsistent('purple circle', mind, signalerType))
        if self.signalIsConsistent(signal, mind, signalerType) and (signal in self.signalSpace):
            possibleSignals = [s for s in self.signalSpace if self.signalIsConsistent(s, mind, signalerType)]
            #rescaledSignalProbability = self.rescaleSignalUtilityForCost(signal, mind)*(1.0/len(possibleSignals))
            probabilityOfSignal = 1.0/len(possibleSignals)
            return(probabilityOfSignal)
        return(0.0)

    def getMindDF(self, mindDictionary):
        mindLabels = list(mindDictionary.keys())
        mindValues = [[v] for v in mindDictionary.values()]
        idx = pd.MultiIndex.from_product(mindValues, names=mindLabels)
        mindCondition = pd.DataFrame(index=idx)
        return(mindCondition)

    def getConditionSignalUtility(self, signal, mindcondition):
        if self.getSignalCost is None:
            return(0)
        return(self.getSignalCost(signal, mindcondition))

    """    
    def rescaleSignalUtilityForCost(self, signal, mind, factorOfDeviationFromUniform = .05):
        signalUtilities = [self.getConditionSignalUtility(s, mind) for s in self.signalSpace]
        signalUtility = self.getConditionSignalUtility(signal, mind)
        rangeMin = min(signalUtilities)
        rangeMax = max(signalUtilities)

        signalSpaceSize = len(self.signalSpace)
        targetMin = 1/signalSpaceSize - factorOfDeviationFromUniform*1/signalSpaceSize
        targetMax = 1/signalSpaceSize + factorOfDeviationFromUniform*1/signalSpaceSize

        if (rangeMax  - rangeMin) == 0:
            return(1)
        else:
            rescaledProbability = ((signalUtility - rangeMin)/(rangeMax  - rangeMin))*(targetMax-targetMin) + targetMin
            #print("signal", signal,"rescaled probability", rescaledProbability, "\n")
            return(rescaledProbability)
    """
