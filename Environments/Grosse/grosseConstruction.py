import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd
import numpy as np

from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace, normalizeValuesPdSeries
import Algorithms.constantNames as NC

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

#multiple goal - Grosse battery example, uniform
def getGoalGivenWorldAndDesire_Grosse(goalSpace, world, desire, goalProbabilities = None):
    goalDict = {NC.INTENTIONS: goalSpace}
    goalSpaceDF = getMultiIndexMindSpace(goalDict)
    
    goalProbabilities = goalSpaceDF.groupby(goalSpaceDF.index.names).apply(lambda x: getConditionGoalPDF_Grosse(goal = x.index.get_level_values(NC.INTENTIONS)[0], world = world))
    goalSpaceDF[NC.P_INTENTION] = goalSpaceDF.index.get_level_values(0).map(normalizeValuesPdSeries(goalProbabilities).get)
    return(goalSpaceDF)


def getConditionGoalPDF_Grosse(goal, world, nullWorld = 'n', nullGoal = 'n'):
    if nullWorld in world:
        assert world == nullWorld, 'incongruous world: cannot have neither battery and a battery value in the world'
    if nullGoal in goal:
        assert goal == nullGoal, "incongruous goal: cannot have both neither battery and a battery value as a goal"
        return(1)
    if goal == 'either':
        return(1)
    return(int(all(g in world for g in goal)))


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


#p(a|w,g) component of the mind - Grosse
"""
    alpha: scalar rationality constant
    actionUtilityFunction: function that takes in (action, world, goal) and returns a scalar utlity
    softmax: boolean; False indicates strict maximization
"""
# Inputs in the callable: actions as tuples indicating agent actions; i.e. (signaler action, receiver action)
class ActionDistributionGivenWorldGoal_Grosse(object):
    def __init__(self, alpha, actionUtilityFunction, softmax=False):
        self.alpha = alpha
        self.getUtilityOfAction = actionUtilityFunction
        self.softmax = softmax
        
    def __call__(self, actionSpace, world, goal):
        #create a dataframe with indices actions
        actionSpaceDF = getMultiIndexMindSpace({NC.ACTIONS:actionSpace})

        #for each action, get the utility given goal, world; transform this into an action distribution
        getConditionActionUtility = lambda x: np.exp(self.alpha*self.getUtilityOfAction(x.index.get_level_values(NC.ACTIONS)[0], world, goal))
        utilities = actionSpaceDF.groupby(actionSpaceDF.index.names).apply(getConditionActionUtility)
        #keep as softmax pdf or transform to strict maximization
        if self.softmax:         
            probabilities = normalizeValuesPdSeries(utilities)
        else:
            maxUtility = max(utilities)
            numberOfOccurances = utilities.value_counts().loc[maxUtility]
            getConditionProbability = lambda x: 1.0/numberOfOccurances if x == maxUtility else 0 
            probabilities = utilities.apply(getConditionProbability)
            
        actionSpaceDF[NC.P_ACTION] = actionSpaceDF.index.get_level_values(0).map(probabilities.get)
        return(actionSpaceDF)

# Grosse action utility - multiple agent actions
"""
    costOfLocation: list of dictionaries: list indices indicate agents [signaler, receiver], dictionaries indicate cost of actions {action key: action cost scalar}
    valueOfReward: scalar reward value for achieving each component of the intended goal
    nullAction: the representation of a null action (default = 'n')
"""
class ActionUtility_Grosse(object):
    def __init__(self, costOfLocation, valueOfReward, nullAction = 'n'):
        self.costOfLocation = costOfLocation
        self.valueOfReward = valueOfReward
        self.nullAction = nullAction

    def __call__(self, action, world, goal):
        assert self.isActionCongruous(action, world), 'action is not possible in this world'
        jointCost  = self.getActionCost(action)
        rewardAmount = self.getReward(action, goal)
        totalUtility = jointCost + rewardAmount
        return(totalUtility)
            
    def isActionCongruous(self, action, world):
        areActionsPossible = [agentAction in world for agentAction in action if agentAction != self.nullAction]
        return(all(areActionsPossible))
    
    #joint cost of action for all agents
    def getActionCost(self, action):
        signalerAction = action[0]
        signalerCost = -abs(self.costOfLocation[0][signalerAction])
        receiverAction = action[1]
        receiverCost = -abs(self.costOfLocation[1][receiverAction])
        jointActionCost = signalerCost + receiverCost
        return(jointActionCost)

    #total reward of action
    def getReward(self, action, goal, nullActionGoal='n'):
        if goal == nullActionGoal:
            return(0)
        if goal == 'either':
            if action == (nullActionGoal, nullActionGoal):
                return(0)
            else:
                return(self.valueOfReward)

        goalList = list(goal)
        reward = 0
        numberItemsInGoal = len(goalList)
        signalerAction = action[0]
        receiverAction = action[1]
        
        if signalerAction in goalList:
            reward += self.valueOfReward/numberItemsInGoal
            goalList.remove(signalerAction)
        if receiverAction in goalList:
            reward += self.valueOfReward/numberItemsInGoal
            goalList.remove(receiverAction)
        return(reward)


class SignalCost_Grosse(object):
    def __init__(self, costMultiplicationFactor=.05):
        self.costMultiplicationFactor = costMultiplicationFactor

    def __call__(self, signal, mind=None, nullSignalSet = ['null', '', 'me']):
        if signal in nullSignalSet:
            return(0)
        signalLength = len(signal)
        cost = -self.costMultiplicationFactor*signalLength
        return(cost)

