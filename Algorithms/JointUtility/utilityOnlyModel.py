import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import warnings

from Environments.Experiment.experimentConstruction import *
import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import GenerateMind


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

###########################################################################################################################################
###########################################################################################################################################

class UtilityDrivenReceiver():
    def __init__(self, signalerLocation, receiverLocation, targetDictionary, valueOfReward, rationality, costFunction = calculateLocationCost_TaxicabMetric, utilityFn = JointActionUtility):
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary
        self.reward = valueOfReward
        self.getCost = costFunction
        self.rationality = rationality
        self.getUtility = utilityFn
        
    def __call__(self, signal):
        getMindFunction = self.setupGetMindFunction()
        condition = self.setupConditionConsistentWithSignal(signal)
        receiverMind = getMindFunction(condition) #consistent Intentions

        actionConsistent = lambda x: 0 if x[NC.ACTIONS][0] != self.signalerLocation else x[NC.P_MIND]
        receiverMind[NC.P_MIND] = receiverMind.reset_index().apply(actionConsistent, axis = 1).tolist()
        receiverMind[NC.P_MIND]= receiverMind[NC.P_MIND]/sum(receiverMind[NC.P_MIND])
        return(receiverMind)

        
    def setupGetMindFunction(self):
        getUtility = self.getUtility(costFunction=self.getCost, 
                                        valueOfReward=self.reward, 
                                        signalerLocation=self.signalerLocation, 
                                        receiverLocation=self.receiverLocation, 
                                        targetDictionary= self.targetDictionary)
        getActionDistribution = ActionDistributionGivenWorldGoal(self.rationality, getUtility)
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
        return(getMind)
    
    def setupConditionConsistentWithSignal(self, signal):        
        intentionSpace = [intention for intention in self.targetDictionary.values() if self.intentionConsistentWithSignal(signal, intention)]
        
        #ensures that if there is nothing consistent with the signal, the receiver can still do something
        if len(intentionSpace) == 0:
            intentionSpace = list(self.targetDictionary.values())
            
        fullActionSpace = getActionSpace_SignalsSeparated(self.targetDictionary, self.signalerLocation, self.receiverLocation)
        condition = {NC.WORLDS: [1], 
                      NC.DESIRES: [1], 
                      NC.INTENTIONS: intentionSpace, 
                      NC.ACTIONS: fullActionSpace}
        return(condition)
        
    def intentionConsistentWithSignal(self, signal, targetState):
        utteranceFeatures = list(signal.split())
        featuresOfTarget = list(targetState.split())
        consistentUtterance = [feature in featuresOfTarget for feature in utteranceFeatures]
        return(all(consistentUtterance))

class UtilityDrivenReceiver_CostRatio():
    def __init__(self, signalerLocation, receiverLocation, targetDictionary, valueOfReward, rationality, receiverCostRatio, costFunction = calculateLocationCost_TaxicabMetric, utilityFn = JointActionUtility_CostRatioReceiver ):
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary
        self.reward = valueOfReward
        self.getCost = costFunction
        self.rationality = rationality
        self.getUtility = utilityFn
        self.receiverCostRatio = receiverCostRatio
        
    def __call__(self, signal):
        getMindFunction = self.setupGetMindFunction()
        condition = self.setupConditionConsistentWithSignal(signal)
        receiverMind = getMindFunction(condition) #consistent Intentions

        actionConsistent = lambda x: 0 if x[NC.ACTIONS][0] != self.signalerLocation else x[NC.P_MIND]
        receiverMind[NC.P_MIND] = receiverMind.reset_index().apply(actionConsistent, axis = 1).tolist()
        receiverMind[NC.P_MIND]= receiverMind[NC.P_MIND]/sum(receiverMind[NC.P_MIND])
        return(receiverMind)

        
    def setupGetMindFunction(self):
        getUtility = self.getUtility(costFunction=self.getCost, 
                                        valueOfReward=self.reward, 
                                        signalerLocation=self.signalerLocation, 
                                        receiverLocation=self.receiverLocation, 
                                        targetDictionary= self.targetDictionary, receiverCostRatio = self.receiverCostRatio)
        getActionDistribution = ActionDistributionGivenWorldGoal(self.rationality, getUtility)
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
        return(getMind)
    
    def setupConditionConsistentWithSignal(self, signal):        
        intentionSpace = [intention for intention in self.targetDictionary.values() if self.intentionConsistentWithSignal(signal, intention)]
        
        #ensures that if there is nothing consistent with the signal, the receiver can still do something
        if len(intentionSpace) == 0:
            intentionSpace = list(self.targetDictionary.values())
            
        fullActionSpace = getActionSpace_SignalsSeparated(self.targetDictionary, self.signalerLocation, self.receiverLocation)
        condition = {NC.WORLDS: [1], 
                      NC.DESIRES: [1], 
                      NC.INTENTIONS: intentionSpace, 
                      NC.ACTIONS: fullActionSpace}
        return(condition)
        
    def intentionConsistentWithSignal(self, signal, targetState):
        utteranceFeatures = list(signal.split())
        featuresOfTarget = list(targetState.split())
        consistentUtterance = [feature in featuresOfTarget for feature in utteranceFeatures]
        return(all(consistentUtterance))
    
class UtilityDrivenSignaler():
    def __init__(self, signalSpace, signalerLocation, receiverLocation, targetDictionary, valueOfReward, rationality, costFunction = calculateLocationCost_TaxicabMetric, signalerInactionPossible=False):
        self.signalSpace = signalSpace
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary

        self.reward = valueOfReward
        self.getCost = costFunction
        self.rationality = rationality
        self.signalerInactionPossible = signalerInactionPossible

    def __call__(self, observation):
        trueTarget = observation[NC.INTENTIONS]
        consistentSignals = [signal for signal in self.signalSpace if self.intentionConsistentWithSignal(signal, trueTarget)]
        numberOfConsistentSignals = len(consistentSignals)
        
        if numberOfConsistentSignals > 0:
            signalerDoProbability, receiverDoProbability, signalerQuitsProbability = self.getAgentActionProbability(trueTarget)
            signalingDictionary = {signal : receiverDoProbability/numberOfConsistentSignals for signal in consistentSignals}
            signalingDictionary[trueTarget] = signalerDoProbability
            
            if signalerQuitsProbability is not None:
                signalingDictionary['quit'] = signalerQuitsProbability

        else: #if no consistent signals then always do for self (or quit)
            if self.signalerInactionPossible:
                signalerDoProbability, signalerQuitsProbability = self.getSignalerDoOrQuitProbability(trueTarget)
                signalingDictionary = {trueTarget: signalerDoProbability, 'quit':signalerQuitsProbability}
            else:
                signalingDictionary = {trueTarget:1.0}

        signalingDF = pd.DataFrame.from_dict(signalingDictionary, orient = 'index').rename(columns={0: NC.PROBABILITY})
        signalingDF.index.name = NC.SIGNALS
        return(signalingDF)

    def getActionUtility(self, action, goal, startingPoint):
        utility = self.getCost(startingPoint, action)
        if self.targetDictionary[action] == goal:
            utility += self.reward
        return(utility)

    def getSignalerDoOrQuitProbability(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.signalerLocation))
        inactionUtility = np.exp(self.rationality*0)
        normalizingConstant = dIYUtility+inactionUtility
        return(dIYUtility/normalizingConstant, inactionUtility/normalizingConstant)

    def getAgentActionProbability(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.signalerLocation))
        receiverDoesUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.receiverLocation))
        
        if self.signalerInactionPossible:
            inactionUtility = np.exp(self.rationality*0)
            normalizingConstant = dIYUtility+receiverDoesUtility+inactionUtility
            return(dIYUtility/normalizingConstant , receiverDoesUtility/normalizingConstant, inactionUtility/normalizingConstant)
        else:
            normalizingConstant = dIYUtility+receiverDoesUtility
            return(dIYUtility/normalizingConstant , receiverDoesUtility/normalizingConstant, None)

    def intentionConsistentWithSignal(self, signal, targetState):
        utteranceFeatures = list(signal.split())
        featuresOfTarget = list(targetState.split())
        consistentUtterance = [feature in featuresOfTarget for feature in utteranceFeatures]
        return(all(consistentUtterance))



class UtilityDrivenSignaler_NoReceiverCosts():
    def __init__(self, signalSpace, signalerLocation, receiverLocation, targetDictionary, valueOfReward, rationality, costFunction = calculateLocationCost_TaxicabMetric):
        self.signalSpace = signalSpace
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary

        self.reward = valueOfReward
        self.getCost = costFunction
        self.rationality = rationality

    def __call__(self, observation):
        trueTarget = observation[NC.INTENTIONS]
        consistentSignals = [signal for signal in self.signalSpace if self.intentionConsistentWithSignal(signal, trueTarget)]
        numberOfConsistentSignals = len(consistentSignals)
        if numberOfConsistentSignals > 0:
            signalerDoProbability, receiverDoProbability = self.getAgentActionProbability(trueTarget)
            signalingDictionary = {signal : receiverDoProbability/numberOfConsistentSignals for signal in consistentSignals}
            signalingDictionary[trueTarget] = signalerDoProbability
        else: #if no consistent signals then always do for self
            signalingDictionary = {trueTarget:1.0}

        signalingDF = pd.DataFrame.from_dict(signalingDictionary, orient = 'index').rename(columns={0: NC.PROBABILITY})
        signalingDF.index.name = NC.SIGNALS
        return(signalingDF)

    def getActionUtility(self, action, goal, startingPoint, receiver = False):
        utility = 0
        if not receiver:
            utility += self.getCost(startingPoint, action)
        if self.targetDictionary[action] == goal:
            utility += self.reward
        return(utility)

    def getAgentActionProbability(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.signalerLocation))
        receiverDoesUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.receiverLocation, receiver=True))
        normalizingConstant = dIYUtility+receiverDoesUtility
        return(dIYUtility/normalizingConstant,receiverDoesUtility/normalizingConstant)

    def intentionConsistentWithSignal(self, signal, targetState):
        utteranceFeatures = list(signal.split())
        featuresOfTarget = list(targetState.split())
        consistentUtterance = [feature in featuresOfTarget for feature in utteranceFeatures]
        return(all(consistentUtterance))

class UtilityDrivenSignaler_RatioReceiverCosts():
    def __init__(self, signalSpace, signalerLocation, receiverLocation, targetDictionary, valueOfReward, rationality, receiverRatioCost, costFunction = calculateLocationCost_TaxicabMetric,signalerInactionPossible=False ):
        self.signalSpace = signalSpace
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.targetDictionary = targetDictionary
        self.receiverRatioCost = receiverRatioCost

        self.reward = valueOfReward
        self.getCost = costFunction
        self.rationality = rationality
        self.signalerInactionPossible = signalerInactionPossible

    def __call__(self, observation):
        trueTarget = observation[NC.INTENTIONS]
        consistentSignals = [signal for signal in self.signalSpace if self.intentionConsistentWithSignal(signal, trueTarget)]
        numberOfConsistentSignals = len(consistentSignals)
        if numberOfConsistentSignals > 0:
            signalerDoProbability, receiverDoProbability,signalerQuitsProbability  = self.getAgentActionProbability(trueTarget)
            signalingDictionary = {signal : receiverDoProbability/numberOfConsistentSignals for signal in consistentSignals}
            signalingDictionary[trueTarget] = signalerDoProbability
            
            if signalerQuitsProbability is not None:
                signalingDictionary['quit'] = signalerQuitsProbability
        else: #if no consistent signals then always do for self
            if self.signalerInactionPossible:
                signalerDoProbability, signalerQuitsProbability = self.getSignalerDoOrQuitProbability(trueTarget)
                signalingDictionary = {trueTarget: signalerDoProbability, 'quit':signalerQuitsProbability}
            else:
                signalingDictionary = {trueTarget:1.0}

        signalingDF = pd.DataFrame.from_dict(signalingDictionary, orient = 'index').rename(columns={0: NC.PROBABILITY})
        signalingDF.index.name = NC.SIGNALS
        return(signalingDF)
    

    def getActionUtility(self, action, goal, startingPoint, receiver = False):
        utility = 0
        if not receiver:
            utility += self.getCost(startingPoint, action)*(1-self.receiverRatioCost)
        else:
            utility += self.getCost(startingPoint, action)*self.receiverRatioCost
        if self.targetDictionary[action] == goal:
            utility += self.reward
        return(utility)

    def getSignalerDoOrQuitProbability(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.signalerLocation))
        inactionUtility = np.exp(self.rationality*0)
        normalizingConstant = dIYUtility+inactionUtility
        return(dIYUtility/normalizingConstant, inactionUtility/normalizingConstant)

    def getAgentActionProbability(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.signalerLocation))
        receiverDoesUtility = np.exp(self.rationality*self.getActionUtility(targetAction, trueGoal, self.receiverLocation, receiver=True ))
        
        if self.signalerInactionPossible:
            inactionUtility = np.exp(self.rationality*0)
            normalizingConstant = dIYUtility+receiverDoesUtility+inactionUtility
            return(dIYUtility/normalizingConstant , receiverDoesUtility/normalizingConstant, inactionUtility/normalizingConstant)
        else:
            normalizingConstant = dIYUtility+receiverDoesUtility
            return(dIYUtility/normalizingConstant , receiverDoesUtility/normalizingConstant, None)
       

    def intentionConsistentWithSignal(self, signal, targetState):
        utteranceFeatures = list(signal.split())
        featuresOfTarget = list(targetState.split())
        consistentUtterance = [feature in featuresOfTarget for feature in utteranceFeatures]
        return(all(consistentUtterance))
