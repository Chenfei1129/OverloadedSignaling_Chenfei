import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import itertools
import pandas as pd
import numpy as np

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace, normalizeValuesPdSeries

########################################
###### Cost Functions
########################################

def calculateLocationCost_TaxicabMetric(agentLocation, proposedActionLocation):
    getDistance = lambda a, b: abs(a-b)
    absoluteActionCosts = [getDistance(locationXY, actionXY) for locationXY, actionXY in zip(agentLocation, proposedActionLocation)]
    totalActionCost = -sum(absoluteActionCosts)
    return(totalActionCost)

def calculateLocationCost_EuclidianMetric(agentLocation, proposedActionLocation):
    getSquaredDistance = lambda a, b: (a-b)**2
    squaredActionCosts = [getSquaredDistance(locationXY, actionXY) for locationXY, actionXY in zip(agentLocation, proposedActionLocation)]
    actionCost = -np.sqrt(sum(squaredActionCosts))
    return(actionCost)

class CalculateLocationCost_TaxicabMetric_receiverCostRatio:
    def __init__(self, receiverCostRatio):
        self.receiverCostRatio = receiverCostRatio
    def __call__(agentLocation, proposedActionLocation, receiver = False):
        if receiver: 
            getDistance = lambda a, b: abs(a-b)
            absoluteActionCosts = [getDistance(locationXY, actionXY) for locationXY, actionXY in zip(agentLocation, proposedActionLocation)]
            totalActionCost = -sum(absoluteActionCosts)*(self.receiverCostRatio)
            return(totalActionCost)
        else:
            getDistance = lambda a, b: abs(a-b)
            absoluteActionCosts = [getDistance(locationXY, actionXY) for locationXY, actionXY in zip(agentLocation, proposedActionLocation)]
            totalActionCost = -sum(absoluteActionCosts)*(1 - self.receiverCostRatio)
            return(totalActionCost)

####################################
##### Action Space Tuples
###################################

def getActionSpace(targetDictionary, signalDictionary, signalerPosition, receiverPosition, signalerInactionPossible = True): 
    targetItemList = list(targetDictionary.keys())
    signalList = list(signalDictionary.keys())

    if signalerInactionPossible:
        signalerActionSpace = targetItemList + signalList + [signalerPosition]
    else:
        signalerActionSpace = targetItemList + signalList
        
    receiverActionSpace = targetItemList + [receiverPosition] 
    jointActionSpace = [(sigAction, recAction) for (sigAction, recAction) in itertools.product(signalerActionSpace,receiverActionSpace) if sigAction != recAction]
    return(jointActionSpace)


######################################
##### Functions for Mind Components
######################################
# P(w) component of the mind - uniform distribution 
def getWorldProbabiltiy_Uniform(worldSpace = [1]):
    uniqueWorldSpace = list(set(worldSpace))
    worldSpaceDF = getMultiIndexMindSpace({NC.WORLDS: uniqueWorldSpace}, [NC.P_WORLD])
    unifProbabilityOfWorld = 1.0/len(uniqueWorldSpace)
    worldSpaceDF[NC.P_WORLD] = worldSpaceDF.groupby(worldSpaceDF.index.names).transform(lambda x: unifProbabilityOfWorld)
    return(worldSpaceDF)

# p(d) component of the mind - uniform distribution
def getDesireProbability_Uniform(desireSpace=[1]):
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


#p(a|w,i) component of the mind 
class ActionDistributionGivenWorldGoal(object):
    def __init__(self, alpha, actionUtilityFunction, softmax=True):
        self.alpha = alpha
        self.getUtilityOfAction = actionUtilityFunction
        self.softmax = softmax
        
    def __call__(self, actionSpace, world, goal):
        #create a dataframe with indices actions
        #actionSpaceDF = getMultiIndexMindSpace({NC.ACTIONS:actionSpace}, [NC.P_ACTION])
        actionSpaceDF = pd.DataFrame.from_dict({NC.ACTIONS:actionSpace})
        actionSpaceDF[NC.P_ACTION] = np.nan
        actionSpaceDF = actionSpaceDF.set_index(NC.ACTIONS)

        #for each action, get the utility given goal, world; transform this into an action distribution
        getConditionActionUtility = lambda x: np.exp(self.alpha*self.getUtilityOfAction(x.index[0], world, goal))
        #actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby(actionSpaceDF.index.names).transform(getConditionActionUtility)
        actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby([NC.ACTIONS]).transform(getConditionActionUtility)
    
        #keep as softmax pdf or transform to strict maximization, normalize
        if self.softmax:
            normalizingConstant = actionSpaceDF[NC.P_ACTION].sum()         
            #actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby(actionSpaceDF.index.names).transform(lambda x: x/normalizingConstant)
            actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby([NC.ACTIONS]).transform(lambda x: x/normalizingConstant)
        else:
            maxUtility = max(actionSpaceDF[NC.P_ACTION])
            numberOfOccurances = actionSpaceDF[NC.P_ACTION].value_counts().loc[maxUtility]
            getConditionProbability = lambda x: np.where(x == maxUtility, 1.0/numberOfOccurances, 0.)
            #actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby(actionSpaceDF.index.names).transform(getConditionProbability)
            actionSpaceDF[NC.P_ACTION] = actionSpaceDF.groupby([NC.ACTIONS]).transform(getConditionProbability)
        return(actionSpaceDF)

######################################
##### Joint Utility
######################################

class JointActionUtility(object):
    def __init__(self, costFunction, valueOfReward, signalerLocation, receiverLocation, targetDictionary):
        self.getCost = costFunction
        self.valueOfReward = valueOfReward
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary

    def __call__(self, action, world, goal):
        #print(goal)
        signalerAction = action[0]
        recevierAction = action[1]
        signalerUtility = self.getUtility(self.signalerLocation, signalerAction, goal)
        receieverUtility = self.getUtility(self.receiverLocation, recevierAction, goal)
        jointUtilityOfAction = receieverUtility + signalerUtility
        return(jointUtilityOfAction)

    def getUtility(self, agentPosition, agentAction, trueGoal):
        actionCost = self.getCost(agentPosition, agentAction)
        #print(trueGoal)
        #print(self.targetDictionary.items() )
        trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]
        getActionReward = lambda action, goalLocation: self.valueOfReward if (action == goalLocation) else 0
        actionReward = getActionReward(agentAction, trueGoalLocation)

        utility = actionCost + actionReward
        return(utility)

class JointActionUtility_CostRatioReceiver2(object):
    def __init__(self, costFunction, valueOfReward, signalerLocation, receiverLocation,  targetDictionary, receiverCostRatio ):
        self.getCost = costFunction
        self.valueOfReward = valueOfReward
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary
        self.receiverCostRatio = receiverCostRatio

    def __call__(self, action, world, goal):
        signalerAction = action[0]
        recevierAction = action[1]
        signalerUtility = self.getUtility(self.signalerLocation, signalerAction, goal, self.receiverCostRatio)
        receieverUtility = self.getUtility(self.receiverLocation, recevierAction, goal, self.receiverCostRatio, receiver=True)
        jointUtilityOfAction = receieverUtility + signalerUtility
        return(jointUtilityOfAction)

    def getUtility(self, agentPosition, agentAction, trueGoal, receiverCostRatio, receiver=False):
        if receiver:
            actionCost = self.getCost(agentPosition, agentAction)*self.receiverCostRatio
        else:
            actionCost = self.getCost(agentPosition, agentAction)*(1 - receiverCostRatio)

        #trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]

        trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]
        getActionReward = lambda action, goalLocation: self.valueOfReward if (action == goalLocation) else 0
        actionReward = getActionReward(agentAction, trueGoalLocation)

        utility = actionCost + actionReward
        return(utility)

class JointActionUtility_CostlessReceiver(object):
    def __init__(self, costFunction, valueOfReward, signalerLocation, receiverLocation, targetDictionary):
        self.getCost = costFunction
        self.valueOfReward = valueOfReward
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary

    def __call__(self, action, world, goal):
        signalerAction = action[0]
        recevierAction = action[1]
        signalerUtility = self.getUtility(self.signalerLocation, signalerAction, goal)
        receieverUtility = self.getUtility(self.receiverLocation, recevierAction, goal, receiver=True)
        jointUtilityOfAction = receieverUtility + signalerUtility
        return(jointUtilityOfAction)

    def getUtility(self, agentPosition, agentAction, trueGoal, receiver=False):
        if receiver:
            actionCost = 0.0
        else:
            actionCost = self.getCost(agentPosition, agentAction)

        #trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]
        trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]
        getActionReward = lambda action, goalLocation: self.valueOfReward if (action == goalLocation) else 0
        actionReward = getActionReward(agentAction, trueGoalLocation)

        utility = actionCost + actionReward
        return(utility)

###
"""
class JointActionUtility_CostRatioReceiver(object):
    def __init__(self, costFunction, valueOfReward, signalerLocation, receiverLocation, targetDictionary):
        self.getCost = costFunction
        self.valueOfReward = valueOfReward
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary

    def __call__(self, action, world, goal, ratio):
        signalerAction = action[0]
        recevierAction = action[1]

        signalerUtility = self.getUtility(self.signalerLocation, signalerAction, goal, ratio)
        receieverUtility = self.getUtility(self.receiverLocation, recevierAction, goal, ratio, receiver=True)
        jointUtilityOfAction = receieverUtility + signalerUtility
        return(jointUtilityOfAction)

    def getUtility(self, agentPosition, agentAction, trueGoal, receiverCostRatio, receiver=False):
        if receiver:
            actionCost = self.getCost(agentPosition, agentAction) * receiverCostRatio
        else:
            actionCost = self.getCost(agentPosition, agentAction) * (1 - receiverCostRatio)

        #trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]

        trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]
        getActionReward = lambda action, goalLocation: self.valueOfReward if (action == goalLocation) else 0
        actionReward = getActionReward(agentAction, trueGoalLocation)

        utility = actionCost + actionReward
        return(utility)
"""
###

#######################
#######################
def getActionSpace_SignalsSeparated(targetDictionary, signalerPosition, receiverPosition): 
    targetItemList = list(targetDictionary.keys())
    receiverMoves = [(signalerPosition, target) for target in targetItemList]
    signalerMoves = [(target, receiverPosition) for target in targetItemList]
    jointActionSpace = signalerMoves + receiverMoves
    return(jointActionSpace)



class AddDoYourselfSignalerUtility():
    def __init__(self, getSignalerUtility, alpha, rewardValue, signalerPosition, targetDictionary, receiverCostRatio = 0 , costFunction = calculateLocationCost_TaxicabMetric,  signalerInactionPossible=False):
        self.getSignalerUtility = getSignalerUtility
        self.alpha = alpha
        self.valueOfReward = rewardValue
        self.signalerPosition = signalerPosition
        self.targetDictionary = targetDictionary
        self.getCost = costFunction
        self.receiverCostRatio = receiverCostRatio
        self.signalerInactionPossible = signalerInactionPossible

    def __call__(self, observation):
        #Returns raw utilities from Imagined We pragmatic signaler
        signalerUtilities = self.getSignalerUtility(observation, returnRawUtilities = True)
        #print("*****")
        #print(signalerUtilities)
        trueGoal = observation[NC.INTENTIONS]
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        #print("Before times alpha DIY")
        #print(self.getUtility(targetAction, trueGoal))
        
        dIYUtility = np.exp(self.alpha*self.getUtility(targetAction, trueGoal))
        
        signalPDF = signalerUtilities.reset_index().set_index(NC.SIGNALS)
        
        signalPDF.loc['do' + str(trueGoal)] = dIYUtility
        
        if self.signalerInactionPossible:
            signalerInactionUtility = np.exp(self.alpha*0)
            signalPDF.loc['quit'] = signalerInactionUtility
        #print(signalPDF)
        signalPDF[NC.UTILITY] = signalPDF[NC.UTILITY]/sum(signalPDF[NC.UTILITY])
        signalPDF = signalPDF.rename(columns={NC.UTILITY: NC.PROBABILITY})
        #print(signalPDF)
        return(signalPDF)


    def getUtility(self, action, goal):
        utility = self.getCost(self.signalerPosition, action)#*(1 - self.receiverCostRatio)
        if self.targetDictionary[action] == goal:
            utility += self.valueOfReward
        return(utility)


class JointActionUtility_CostRatioReceiver(object):
    def __init__(self, costFunction, valueOfReward, signalerLocation, receiverLocation,  targetDictionary, receiverCostRatio ):
        self.getCost = costFunction
        self.valueOfReward = valueOfReward
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary
        self.receiverCostRatio = receiverCostRatio

    def __call__(self, action, world, goal):
        signalerAction = action[0]
        recevierAction = action[1]
        signalerUtility = self.getUtility(self.signalerLocation, signalerAction, goal, self.receiverCostRatio)
        receieverUtility = self.getUtility(self.receiverLocation, recevierAction, goal, self.receiverCostRatio, receiver=True)
        jointUtilityOfAction = receieverUtility + signalerUtility
        return(jointUtilityOfAction)

    def getUtility(self, agentPosition, agentAction, trueGoal, receiverCostRatio, receiver=False):
        if receiver:
            actionCost = self.getCost(agentPosition, agentAction)*self.receiverCostRatio
        else:
            actionCost = self.getCost(agentPosition, agentAction)*(1 - receiverCostRatio)

        #trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]

        trueGoalLocation = [location for location, features in self.targetDictionary.items() if features == trueGoal][0]
        getActionReward = lambda action, goalLocation: self.valueOfReward if (action == goalLocation) else 0
        actionReward = getActionReward(agentAction, trueGoalLocation)

        utility = actionCost + actionReward
        return(utility)
'''

#######################
#######################
def getActionSpace_SignalsSeparated(targetDictionary, signalerPosition, receiverPosition): 
    targetItemList = list(targetDictionary.keys())
    receiverMoves = [(signalerPosition, target) for target in targetItemList]
    signalerMoves = [(target, receiverPosition) for target in targetItemList]
    jointActionSpace = signalerMoves + receiverMoves
    return(jointActionSpace)



class AddDoYourselfSignalerUtility():
    def __init__(self, getSignalerUtility, alpha, rewardValue, signalerPosition, targetDictionary, costFunction = calculateLocationCost_TaxicabMetric, signalerInactionPossible=False):
        self.getSignalerUtility = getSignalerUtility
        self.alpha = alpha
        self.valueOfReward = rewardValue
        self.signalerPosition = signalerPosition
        self.targetDictionary = targetDictionary
        self.getCost = costFunction
        self.signalerInactionPossible = signalerInactionPossible

    def __call__(self, observation):
        #Returns raw utilities from Imagined We pragmatic signaler
        signalerUtilities = self.getSignalerUtility(observation, returnRawUtilities = True)
        trueGoal = observation[NC.INTENTIONS]
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal]
        dIYUtility = np.exp(self.alpha*self.getUtility(targetAction, trueGoal))

        signalPDF = signalerUtilities.reset_index().set_index(NC.SIGNALS)
        signalPDF.loc[trueGoal] = dIYUtility

        if self.signalerInactionPossible:
            signalerInactionUtility = np.exp(self.alpha*0)
            signalPDF.loc['quit'] = signalerInactionUtility

        signalPDF[NC.UTILITY] = signalPDF[NC.UTILITY]/sum(signalPDF[NC.UTILITY])
        signalPDF = signalPDF.rename(columns={NC.UTILITY: NC.PROBABILITY})
        return(signalPDF)

    def getUtility(self, action, goal):
        utility = self.getCost(self.signalerPosition, action)
        if self.targetDictionary[action] == goal:
            utility += self.valueOfReward
        return(utility)
'''
