import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import itertools
import pandas as pd
import numpy as np

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace, normalizeValuesPdSeries

#Misyak construction of common ground
def getWorldSpace(wall, nBoxes, nRewards): #omega world, list output
    if wall:
        possibleWorlds = [w for w in itertools.product([1,0], repeat = nBoxes)] 
    else:
        possibleWorlds = [w for w in itertools.product([1,0], repeat = nBoxes) if list(w).count(1) == nRewards]
    return(possibleWorlds)

def getActionSpace(nBoxes, nReceiverChoices): #omega actions, list output
    possibleActions = [a for a in itertools.product([1,0], repeat = nBoxes) 
                       if sum(a) <= nReceiverChoices]
    return(possibleActions)

def getSignalSpace(nBoxes, nSignals): #omega singals, list output
    all_utterances = [c for c in itertools.product([1,0], repeat = nBoxes) 
                      if (sum(c) <= nSignals)] 
    return(all_utterances)


#Mind components
# P(w) component of the mind - uniform distribution 
def getWorldProbabiltiy_Uniform(worldSpace):
    uniqueWorldSpace = list(set(worldSpace))
    worldSpaceDF = getMultiIndexMindSpace({NC.WORLDS: uniqueWorldSpace}, [NC.P_WORLD])
    unifProbabilityOfWorld = 1.0/len(uniqueWorldSpace)
    worldSpaceDF[NC.P_WORLD] = worldSpaceDF.groupby(worldSpaceDF.index.names).transform(lambda x: unifProbabilityOfWorld)
    return(worldSpaceDF)

# p(d) component of the mind - uniform distribution
def getDesireProbability_Uniform(desireSpace):
    uniqueDesireSpace = list(set(desireSpace))
    desireSpaceDF = getMultiIndexMindSpace({NC.DESIRES: uniqueDesireSpace}, [NC.P_DESIRE])
    unifProbabilityOfDesire = 1.0/len(uniqueDesireSpace)
    desireSpaceDF[NC.P_DESIRE] = desireSpaceDF.groupby(desireSpaceDF.index.names).transform(lambda x: unifProbabilityOfDesire)
    return(desireSpaceDF)

#p(i|w,d) component of the mind
def getGoalGivenWorldAndDesire_Uniform(intentionSpace, world, desire):
    uniqueIntentionSpace = list(set(intentionSpace))
    intentionSpaceDF = getMultiIndexMindSpace({NC.INTENTIONS: uniqueIntentionSpace}, [NC.P_INTENTION])
    unifProbabilityOfIntention = 1.0/len(uniqueIntentionSpace)
    intentionSpaceDF[NC.P_INTENTION] = intentionSpaceDF.groupby(intentionSpaceDF.index.names).transform(lambda x: unifProbabilityOfIntention)
    return(intentionSpaceDF)


#p(a|w,i) component of the mind - Misyak
"""
    alpha: scalar rationality constant
    actionUtilityFunction: function that takes in (action, world, goal) and returns a scalar utlity
    softmax: boolean; False indicates strict maximization
"""
class ActionDistributionGivenWorldGoal(object):
    def __init__(self, alpha, actionUtilityFunction, softmax=False):
        self.alpha = alpha
        self.getUtilityOfAction = actionUtilityFunction
        self.softmax = softmax
        
    def __call__(self, actionSpace, world, goal):
        #create a dataframe with indices actions
        actionSpaceDF = getMultiIndexMindSpace({NC.ACTIONS:actionSpace}, [NC.P_ACTION])

        #for each action, get the utility given goal, world; transform this into an action distribution
        getConditionActionUtility = lambda x: np.exp(self.alpha*self.getUtilityOfAction(x.index.get_level_values(NC.ACTIONS)[0], world))
        actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby(actionSpaceDF.index.names).transform(getConditionActionUtility)
    
        #keep as softmax pdf or transform to strict maximization, normalize
        if self.softmax:
            normalizingConstant = actionSpaceDF[NC.P_ACTION].sum()         
            actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby(actionSpaceDF.index.names).transform(lambda x: x/normalizingConstant)
        else:
            maxUtility = max(actionSpaceDF[NC.P_ACTION])
            numberOfOccurances = actionSpaceDF[NC.P_ACTION].value_counts().loc[maxUtility]
            getConditionProbability = lambda x: np.where(x == maxUtility, 1.0/numberOfOccurances, 0.)
            actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby(actionSpaceDF.index.names).transform(getConditionProbability)
        return(actionSpaceDF)


#Utilities
# utility of action U(a,w,g) - misyak
"""
    costOfLocation: int/float or iterable. If int, a fixed cost associated with the number of actions taken. If iterable, the cost of each action in each location
    valueOfReward = int/float. The value of each reward an action taken receives
    costOfNonReward = int/float. The cost associated with each locaiton action that does not result in reward
"""
class ActionUtility(object):
    def __init__(self, costOfLocation, valueOfReward, costOfNonReward):
        self.costOfLocation = costOfLocation
        self.valueOfReward = valueOfReward
        self.costOfNonReward = costOfNonReward

    def __call__(self, action, world, goal=None):
        numberOfLocationsInWorld = len(world)
        locationActionCost = self.getLocationCostList(numberOfLocationsInWorld)
        locationRewardValue = [self.valueOfReward if location == 1 else 0.0 for location in world]
        locationNonRewardCost = [-abs(self.costOfNonReward) if location == 0 else 0.0 for location in world]

        totalLocationValue = [sum((costAct,costNoReward,valueReward)) for costAct, costNoReward, valueReward in zip(locationActionCost, locationNonRewardCost, locationRewardValue)]
        utilityOfAction = [actionValue if action == 1 else 0 for actionValue, action in zip(totalLocationValue,action)]
        return(sum(utilityOfAction))

    def getLocationCostList(self, numberLocations):
        if isinstance(self.costOfLocation, int) or isinstance(self.costOfLocation, float):
            locationCost = [-abs(self.costOfLocation)]*numberLocations
        else:
            assert len(self.costOfLocation) == numberLocations, "Location cost must be either an int/float or iterable of world length"
            locationCost = [-abs(locCost) for locCost in self.costOfLocation]
        return(locationCost)


"""
    pass in a signal and outout the scalar cost of that signal 
    based on how many token/pieces of information used to convey that signal and their respective costs
    default signal marker (indication of a signal in a location) = 1
    default null signal marker (indication of no signal in a location ) = 0
"""
class SignalCost_Misyak(object):
    def __init__(self, signalCosts, signalMarker = 1, nullMarker = 0):
        self.signalCosts = signalCosts
        self.signalMarker = signalMarker
        self.nullMarker = nullMarker

    def __call__(self, signal, mind=None):
        assert (signal.count(self.signalMarker) + signal.count(self.nullMarker))== len(signal), "signal contains undefined signal at some location"

        if isinstance(self.signalCosts, int) or isinstance(self.signalCosts, float):
            numberOfTokensUsed = signal.count(self.signalMarker)
            totalSignalCost = -abs(self.signalCosts)*numberOfTokensUsed
        else:
            totalSignalCost = sum([-abs(locCost) for locSignal, locCost in zip(signal, self.signalCosts) if locSignal == self.signalMarker])
        return(totalSignalCost)


"""
    constructed with a function defining costs and another for action utility.
    Takes in a signal and a pandas dataframe row with indices defining components of the mind
    Outputs the sinal utility as c(signal) + u(action) = cost(signal) + cost(action, world) + reward(action, world, goal)
"""
class SignalUtility(object):
    def __init__(self, signalCostFunction, actionUtilityFunction):
        self.getSignalCost = signalCostFunction
        self.getActionUtility = actionUtilityFunction

    def __call__(self, signal, mind):
        action = mind.index.get_level_values(NC.ACTIONS)[0]
        world =  mind.index.get_level_values(NC.WORLDS)[0]
        intention =  mind.index.get_level_values(NC.INTENTIONS)[0]
        actionUtility = self.getActionUtility(action, world, intention)

        signalCost = self.getSignalCost(signal)
        signalUtility = actionUtility + signalCost
        return(signalUtility)


