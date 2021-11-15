import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np 
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.OverloadedReceiver import ReceiverZero
from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace

class SignalerOne(object):
    def __init__(self, alpha, signalSpace, getActionUtility, getReceiverZero, signalIsConsistent=None, getSignalCost = None):
        self.alpha = alpha
        self.signalSpace = signalSpace
        self.getActionUtility = getActionUtility
        self.getReceiverZero = getReceiverZero
        self.getSignalCost = getSignalCost
        self.signalIsConsistent = signalIsConsistent

    def __call__(self, observation, returnRawUtilities = False):
        #print(observation)
        signalSpaceDF = getMultiIndexMindSpace({NC.SIGNALS: self.signalSpace})
        #print(signalSpaceDF)

        #get the signal utility for each signal with respect to the observation from environment
        getConditionUtility = lambda x: np.exp(self.alpha*self.getUtilityofSignal(observation, x.index.get_level_values(NC.SIGNALS)[0]))
        utilities = signalSpaceDF.groupby(signalSpaceDF.index.names).apply(getConditionUtility)
        
        if returnRawUtilities:
            return(pd.DataFrame(utilities).rename(columns={0: NC.UTILITY}))
        else:
            sumOfUtilities = sum(utilities)
            probabilities = utilities.groupby(utilities.index.names).apply(lambda x: x/sumOfUtilities)

            signalSpaceDF['probability'] = signalSpaceDF.index.get_level_values(0).map(probabilities.get)
            #print(signalSpaceDF)
        
            return(signalSpaceDF)  

    def getUtilityofSignal(self, observation, signal):
        #determine which mind components are observed
        if NC.INTENTIONS in observation.keys():
            intention = observation[NC.INTENTIONS]
        else:
            intention = None

        if NC.WORLDS in observation.keys():
            world = observation[NC.WORLDS]
        else:
            world = None

        #get the posterior of the mind, sum across all possible minds to get a distribution of actions
        mindPosterior = self.getReceiverZero(signal)
        actionPosterior = pd.DataFrame(mindPosterior.groupby(level=[NC.ACTIONS]).sum())
        #if signal == 'purple circle':
            #print(mindPosterior)

        #find the action utilities and evaluate with respect to information from the speaker (observation) E_a[U(mind_speaker, a)|signal]
        
        # If we want to limit ourselves to only consistent signals
        """        
        if self.signalIsConsistent: THIS NEEDS TO BE FIXED
            signalConsistent = lambda y: self.signalIsConsistent(signal, {NC.INTENTIONS: y.index.get_level_values(NC.INTENTIONS)[0], NC.ACTIONS:y.index.get_level_values(NC.ACTIONS)[0]})
            signalConsistent = self.signalIsConsistent(signal, {NC.INTENTIONS: intention})
            getConditionLikelihood = lambda x: signalLikelihoods[x.index.get_level_values(NC.INTENTIONS)[0]] if signalConsistent(x) else 0 
            getConditionActionUtility = lambda x: self.getActionUtility(x.index.get_level_values(NC.ACTIONS)[0], world, intention) if signalConsistent(x) else 0 
            
        else:
        """
        getConditionActionUtility = lambda x: self.getActionUtility(x.index.get_level_values(NC.ACTIONS)[0], world, intention)

        actionPosterior[NC.UTILITY] = actionPosterior.groupby(actionPosterior.index.names).apply(getConditionActionUtility)
        expectedSignalReward = sum(actionPosterior[NC.P_MINDPOSTERIOR]*actionPosterior[NC.UTILITY])
        
        
            
        #signal cost
        signalCost = -abs(self.getCostOfSignalFromFunction(signal))

        totalUtility =  expectedSignalReward + signalCost

        return(totalUtility)

    def getCostOfSignalFromFunction(self, signal):
        if self.getSignalCost is None:
            return(0)
        return(self.getSignalCost(signal))
