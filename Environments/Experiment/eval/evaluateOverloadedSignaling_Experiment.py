import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import seaborn as sns

sys.path.append('../src/')
sys.path.append('../envs/experiment/')

from consistentSignalChecks_Experiment import SignalIsConsistent_Experiment

from GenerativeSignaler import SignalerZero
from OverloadedReceiver import ReceiverZero
from OverloadedSignaler import SignalerOne
from experimentConstruction import *
from mindConstruction import *
import namingConstants as NC

class IWTrialSampler(object):
    def __init__(self, conditionParameters, fixedParameters):
        self.conditionParameters = conditionParameters
        self.fixedParameters = fixedParameters
        
    def __call__(self, trueGoal, costLevel, alpha, nreps):  
        getReceiver, getSignaler = self.getSignalerAndReceiverInferenceFunctions(costLevel, alpha)
        signalerActionPDF = getSignaler({NC.INTENTIONS: trueGoal})
        
        recordedTrialElements = {'sAchievesTarget':0, 'rAchievesTarget':0}
        
        costDictionary = self.translateCostSchemeToInputs(costLevel)    
        trueGoalLocation = [loc for loc, features in costDictionary['targetDictionary'].items() if features == trueGoal][0]
        agentAchievesGoal = lambda trueGoalLoc, action: trueGoalLoc == action
        
        for i in range(nreps): 
            (signalerAction, receiverAction) = self.takeSingleJointSample(signalerActionPDF, getReceiver, costDictionary, trueGoal)
            if agentAchievesGoal(trueGoal, signalerAction):
                recordedTrialElements['sAchievesTarget'] = recordedTrialElements['sAchievesTarget'] + 1
            if agentAchievesGoal(trueGoalLocation, receiverAction):
                recordedTrialElements['rAchievesTarget'] = recordedTrialElements['rAchievesTarget'] + 1
        
        recordedTrialElements['sAchievesTargetProp'] = recordedTrialElements['sAchievesTarget']/nreps
        recordedTrialElements['rAchievesTargetProp'] = recordedTrialElements['rAchievesTarget']/nreps
        recordedTrialElements['targetAchievedProp'] = (recordedTrialElements['sAchievesTarget']+recordedTrialElements['rAchievesTarget'])/nreps
        print("condition alpha", alpha, "cost", costLevel)
        return(pd.Series(recordedTrialElements))
        
    ##############################################
    ##### Inference Helpers
    ###############################################        
    def getSignalerAndReceiverInferenceFunctions(self, costs, alpha):
        costDictionary = self.translateCostSchemeToInputs(costs)
        helperFunctions = self.generateHelpersForInference(costDictionary, alpha)
        getReceiverZero, getSignalerOne = self.generateInferenceFunctions(helperFunctions, costs, alpha)
        return([getReceiverZero, getSignalerOne])        
        
    def translateCostSchemeToInputs(self, costs): #Running
        costTuple = self.conditionParameters[costs]
        signalerLocation, receiverLocation, signalLocs, targetLocs, valueOfReward = costTuple

        possibleSignals = self.conditionParameters['signals']
        signalDictionary = {signalLoc : signalFeature for signalLoc, signalFeature in zip(signalLocs, possibleSignals)}

        possibleTargets = self.conditionParameters['targets']
        targetDictionary = {targetLoc: targetFeature for targetLoc, targetFeature in zip(targetLocs, possibleTargets)}

        return({'signalerLocation': signalerLocation, 
                'receiverLocation':receiverLocation, 
                'signalDictionary':signalDictionary, 
                'targetDictionary':targetDictionary, 
                'valueOfReward': valueOfReward})

    def generateHelpersForInference(self, costDictionary, beta):
        getCost = calculateLocationCost_TaxicabMetric
        signalIsConsistent = SignalIsConsistent_Experiment(costDictionary['signalDictionary'], 
                                                           costDictionary['targetDictionary'], 
                                                           costDictionary['signalerLocation'], 
                                                           costDictionary['receiverLocation'])
        getUtility = JointActionUtility(costFunction=getCost, 
                                    valueOfReward=costDictionary['valueOfReward'], 
                                    signalerLocation=costDictionary['signalerLocation'], 
                                    receiverLocation=costDictionary['receiverLocation'], 
                                    targetDictionary=costDictionary['targetDictionary'], 
                                    signalDictionary=costDictionary['signalDictionary'])

        getActionDistribution = ActionDistributionGivenWorldGoal(beta, getUtility)
        getMind = GenerateMind(getWorldProbabiltiy_Uniform, 
                               getDesireProbability_Uniform, 
                               getGoalGivenWorldAndDesire_Uniform, 
                               getActionDistribution)
        return({'F(consistency)':signalIsConsistent, 
                'F(Utility)':getUtility, 
                'F(a|w,i)':getActionDistribution, 
                'F(mind)':getMind})
    
    def generateInferenceFunctions(self, inferenceHelperDictionary, costs, alpha):
        signalSpace = self.conditionParameters['signals'] + self.conditionParameters['targets'] ######## Check
        signalIsConsistent = inferenceHelperDictionary['F(consistency)']
        
        conditionDictionary = self.getConditionDictionary(costs, self.fixedParameters)
        getMind = inferenceHelperDictionary['F(mind)']
        signalCategoryPrior = {'1':1.0}
        getUtility = inferenceHelperDictionary['F(Utility)']
        
        getGenerativeSignaler = SignalerZero(signalSpace, signalIsConsistent)
        getReceiverZero = ReceiverZero(commonGroundDictionary=conditionDictionary, constructMind=getMind, getSignalerZero=getGenerativeSignaler, signalCategoryPrior=signalCategoryPrior)
        getSignalerOne = SignalerOne(alpha, signalSpace, getUtility, getReceiverZero)
        return([getReceiverZero, getSignalerOne])
    
    def getConditionDictionary(self, costs, mindSpace):  
        costDictionary = self.translateCostSchemeToInputs(costs)
        
        signalerLocation = costDictionary['signalerLocation']
        receiverLocation = costDictionary['receiverLocation']
        signalDictionary = costDictionary['signalDictionary']
        targetDictionary = costDictionary['targetDictionary']
        actionSpace = getActionSpace(targetDictionary, signalDictionary, signalerLocation, receiverLocation)
        
        mindSpace[NC.ACTIONS] = actionSpace
        return(mindSpace)

    ##############################################
    ##### Action Samplers 
    ###############################################  
    def takeSingleJointSample(self, signalerActionPDF, getReceiverFunction, costDictionary, goal):
        receiverLocation = costDictionary['receiverLocation']        
        
        sampledSignalerAction = self.sampleSignalerResponse(signalerActionPDF)
        
        if sampledSignalerAction == goal:
            receiverAction = receiverLocation
        else:
            receiverActionPDF = getReceiverFunction(sampledSignalerAction)
            receiverAction = self.sampleReceiverAction(receiverActionPDF)
        return((sampledSignalerAction, receiverAction))

    
    def sampleSignalerResponse(self, signalerActionPDFDataframe):
        sampledRow = signalerActionPDFDataframe.sample(weights = signalerActionPDFDataframe['probabilities'])
        signalString = sampledRow.index.get_level_values(NC.SIGNALS)[0]
        return(signalString)

    def sampleReceiverAction(self, receiverMindPosterior): # FIX
        sampledRow = receiverMindPosterior.sample(weights = receiverMindPosterior[NC.P_MINDPOSTERIOR])
        jointActionTuple = sampledRow.index.get_level_values(NC.ACTIONS)[0]
        receiverAction = jointActionTuple[1]
        return(receiverAction)

    def totalAchievedUtility(self):
        pass


def drawPerformanceStackPlot(dfCondition, axLabel, axForDraw):
    x = list(dfCondition.index.get_level_values('alpha'))
    y_signalerAchievesGoal = dfCondition['sAchievesTargetProp'].values
    y_receiverAchievesGoal = dfCondition['rAchievesTargetProp'].values
    stackedPlot = axForDraw.stackplot(x,y_signalerAchievesGoal,y_receiverAchievesGoal, labels=['Signaler','Receiver'], colors = ['tab:green', 'tab:olive'], alpha = .5)
    axForDraw.set_yticks([0,.5,1])
    axForDraw.set_xticks([0,.5, 1, 1.5,2])