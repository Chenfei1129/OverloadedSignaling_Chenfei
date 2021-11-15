import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))

import pandas as pd
import warnings

import Algorithms.constantNames as NC
import Simulations.modelLabels as ML


class SimulateModelFromDF():
    def __init__(self, buildModels, getUtility, receiverSpecificUtility = None, signalerCanQuit = False):
        self.iterCounter = 0
        self.setupModelInferences = buildModels
        self.getIndividualUtility = getUtility 
        # For the costless receiver
        self.receiverSpecificUtility = receiverSpecificUtility
        self.signalerCanQuit = signalerCanQuit

    def __call__(self, dfRow):
        modelConditionDictionary = self.setupModelInferences(environmentParameters={'signals': dfRow[ML.SIGNAL_SPACE], 
                                                                                    'targetDictionary': dfRow[ML.TARGET_DICT], 
                                                                                    'signalerLocation': dfRow[ML.S_LOCATION], 
                                                                                    'receiverLocation': dfRow[ML.R_LOCATION]})
        
        goalLocation = [loc for loc, item in dfRow[ML.TARGET_DICT].items() if item == dfRow[ML.INTENTION]][0]
        utilities = self.getWeModelUtilities(dfRow[ML.S_LOCATION], dfRow[ML.R_LOCATION], goalLocation)

        for modelName, modelInference in modelConditionDictionary.items():
            print("model: ", modelName, "\n")
            getSignaler, getReceiver = modelInference#
            formattedTrueGoal = self.formatIntention(modelName, dfRow[ML.INTENTION])
            signalerChoice = self.sampleMaxFromPDF(getSignaler, formattedTrueGoal)#
            print("signaler choice: ", signalerChoice, "\n")
            if self.signalerSendsSignal(signalerChoice, dfRow[ML.SIGNAL_SPACE]):
                receiverActionChoice = self.sampleReceiverChoice(getReceiver, signalerChoice, modelName, dfRow[ML.TARGET_DICT])
                print('goal loc', goalLocation,'receiver choice: ', receiverActionChoice, "\n")
            else:
                receiverActionChoice = None
                
            trialMetrics = self.getModelMetrics(signalerChoice, receiverActionChoice, modelName, dfRow)
            utilities.update(trialMetrics)

        self.iterCounter += 1
        print('Run: ', self.iterCounter)
        output = self.formatOutput(utilities)
        with open('progress.log','a+') as prog_log:
            prog_log.write(str(output)+'\n')
        print(output)
        return(output)
    
    def getModelMetrics(self, signalerChoice, receiverActionChoice, modelName, dfRow):
        trueGoal = dfRow[ML.INTENTION]
        targetDict = dfRow[ML.TARGET_DICT]
        sig = dfRow[ML.S_LOCATION]
        rec = dfRow[ML.R_LOCATION]
        goalLocation = [loc for loc, item in targetDict.items() if item == trueGoal][0]
        #print('signal choice')
        #signalerChoice = signalerChoice[2:]
        #print(signalerChoice)
        signalerQuits = True if signalerChoice == 'quit' else False
        signalerChoiceDo = signalerChoice[2:]
        signalerAchievesGoal = True if trueGoal == signalerChoiceDo else False

       # print(signalerQuits)
        if signalerQuits:
            print("SIGNALER QUITS")
            if not self.signalerCanQuit:
                warnings.warn("model signaler chooses quit, but quit is not enabled for CentralControl/DIY Signaler")

        if signalerAchievesGoal or signalerQuits or receiverActionChoice == rec:
            receiverIntentionChoice = None
            receiverAchievesGoal = False
        else:
            #print(targetDict.items())
            #print(receiverActionChoice)
            receiverIntentionChoice = [item for loc, item in targetDict.items() if loc == receiverActionChoice][0]
            receiverAchievesGoal = True if trueGoal == receiverIntentionChoice else False 
        
        goalAchieved = True if any([signalerAchievesGoal, receiverAchievesGoal]) else False
        
        #####################################################################################################3
        if signalerAchievesGoal:
            signalerActionChoice = [loc for loc, item in targetDict.items() if item == signalerChoiceDo][0]
            utility = self.getIndividualUtility(sig, signalerActionChoice, goalLocation)
        elif signalerQuits:
            utility = 0
        else:
            if self.receiverSpecificUtility is not None:
                utility = self.receiverSpecificUtility(rec, receiverActionChoice, goalLocation)
            else:
                utility = self.getIndividualUtility(rec, receiverActionChoice, goalLocation)
        #####################################################################################################3

        modelMetrics = [signalerChoice, receiverIntentionChoice, signalerAchievesGoal, receiverAchievesGoal, goalAchieved, utility]
        metricNames = ML.SHARED_METRICS
        modelMetricNames = [modelName +"_"+ label for label in metricNames]
        metricDictionary = {metricName: metric for metricName, metric in zip(modelMetricNames, modelMetrics)}
        return(metricDictionary)
        
    def sampleSignalerChoice(self, getSignalPDF, trueGoal):
        signalPdf = getSignalPDF(trueGoal)
        signal = signalPdf.sample(weights = signalPdf.columns[0]).index[0]
        print(signal)
        return(signal)
    
    def sampleMaxFromPDF(self, getSignalPDF, trueGoal):
        pdfToSample = getSignalPDF(trueGoal)        
        maxProbability = pdfToSample.max().values[0]-10E-7
        maxDF = pdfToSample.loc[pdfToSample[pdfToSample.columns[0]] >= maxProbability]
        #print(maxDF)
        sampledValue = maxDF.sample().index[0]
        return(sampledValue)
    
    def sampleReceiverChoice(self, getReceiverPDF, signal, modelName, targetDict):
        receiverPDF = getReceiverPDF(signal)
        if sum(receiverPDF[receiverPDF.columns[0]]) == 0:
          warnings.warn("signaler signal not consistent with anything, receiver samples randomly.")
          receiverChoice = receiverPDF.sample()
        else:
          receiverChoice = receiverPDF.sample(weights = receiverPDF.columns[0])
        
        receiverModelsMind = self.doesReceiverModelIncludeMind(modelName)
        if receiverModelsMind:
            receiverAction = receiverChoice.index.get_level_values(NC.ACTIONS)[0][1]
        else: 
            sampledIntention = receiverChoice.index.get_level_values(NC.INTENTIONS)[0]
            receiverAction = [loc for loc, item in targetDict.items() if item == sampledIntention][0]
        return(receiverAction)
    
    # Format goal input for model type: RSA models take in the intention rather than observation dictionary 
    def formatIntention(self, modelName, rawIntention):
        if 'RSA' in modelName:
            intention = rawIntention
        else:
            intention = {NC.INTENTIONS:rawIntention}
        return(intention)
    
    # Check the format of the receiver output: RSA Listener models output intention only - instead of the entire mind
    def doesReceiverModelIncludeMind(self, modelName):
        return('RSA' not in modelName)
    
    #Checks if the signaler sends a signal or does for self
    def signalerSendsSignal(self, signalerAction, signalSpace):
        return(signalerAction in signalSpace)
    
    def getWeModelUtilities(self, signalerLoc, receiverLoc, truGoalLocation):
        signalerUtility = self.getIndividualUtility(signalerLoc, truGoalLocation, truGoalLocation)
        if self.receiverSpecificUtility is not None:
            receiverUtility = self.receiverSpecificUtility(receiverLoc, truGoalLocation, truGoalLocation)
        else:
            receiverUtility = self.getIndividualUtility(receiverLoc, truGoalLocation, truGoalLocation)

        if self.signalerCanQuit:
            utilities = [signalerUtility, receiverUtility, 0]
            actor = ['signaler', 'receiver', 'quit'][utilities.index(max(utilities))]
            DIYSignaler = max([signalerUtility, 0])
            basicMetrics = {'CentralControl_utility': max(utilities), 'CentralControl_actor':actor, 'DIYSignaler_utility': DIYSignaler}
        else:
            utilities = [signalerUtility, receiverUtility]
            actor = ['signaler', 'receiver'][utilities.index(max(utilities))]
            basicMetrics = {'CentralControl_utility': max(utilities), 'CentralControl_actor':actor, 'DIYSignaler_utility': signalerUtility}
        return(basicMetrics)
        
    def formatOutput(self, completeModelDictionary):
        comparisonModels = self.setupModelInferences.modelNames
        metricNames = ML.SHARED_METRICS
        newColumns = [model + "_" + metricName for model in comparisonModels for metricName in metricNames] + ['CentralControl_utility', 'CentralControl_actor', 'DIYSignaler_utility']
        outputList = [completeModelDictionary[col] for col in newColumns]
        return(pd.Series(outputList, index=newColumns))
        



###################################################################################################################################
################ Utility construction - callable (agentLocation, action, trueGoalLocation)
###################################################################################################################################
class SingleAgentUtility_CustomCostFunction():
    def __init__(self, reward, getCostFunction):
        self.reward = reward
        self.getCost = getCostFunction

    def __call__(self, agentLocation, action, trueGoalLocation):
        getActionReward = lambda act, goalLoc: self.reward if act == goalLoc else 0
        
        utility = self.getCost(agentLocation, action) + getActionReward(action, trueGoalLocation)
        return(utility)
    
class SingleAgentUtility_TaxicabMetric_Ratio():
    def __init__(self, reward, ratioCost, singleStepCost = 1):
        self.reward = reward
        self.ratioCost = ratioCost
        self.singleStepCost = abs(singleStepCost)

    def __call__(self, agentLocation, action, trueGoalLocation):
        getActionReward = lambda act, goalLoc: self.reward if act == goalLoc else 0
        
        getDistance = lambda a, b: abs(a-b)
        getCost = lambda startCoords, endCoords: -sum([getDistance(p, q) for p, q in zip(startCoords, endCoords)])*self.singleStepCost
        
        utility = getCost(agentLocation, action)*self.ratioCost + getActionReward(action, trueGoalLocation)
        return(utility)
    
class SingleAgentUtility_TaxicabMetric():
    def __init__(self, reward, singleStepCost = 1):
        self.reward = reward
        self.singleStepCost = abs(singleStepCost)

    def __call__(self, agentLocation, action, trueGoalLocation):
        getActionReward = lambda act, goalLoc: self.reward if act == goalLoc else 0
        
        getDistance = lambda a, b: abs(a-b)
        getCost = lambda startCoords, endCoords: -sum([getDistance(p, q) for p, q in zip(startCoords, endCoords)])*self.singleStepCost
        
        utility = getCost(agentLocation, action) + getActionReward(action, trueGoalLocation)
        return(utility)

class CostlessReceiverUtility():
    def __init__(self, reward):
        self.reward = reward
    def __call__(self, agentLocation, action, trueGoalLocation):
        getActionReward = lambda act, goalLoc: self.reward if act == goalLoc else 0
        utility =  getActionReward(action, trueGoalLocation)
        return(utility)
