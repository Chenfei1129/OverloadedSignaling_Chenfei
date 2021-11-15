import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd 
import numpy as np
import warnings

import Algorithms.constantNames as NC

class SpeakerActionSignalDistribution():
    def __init__(self, getPragmaticSpeaker, getPragmaticReceiver, targetDictionary, signalerLocation, receiverLocation, rationality, valueOfReward, getActionCost, signalerInactionPossible=False):
        self.getPragmaticSpeaker = getPragmaticSpeaker
        self.getPragmaticReceiver = getPragmaticReceiver

        self.rationality = rationality
        self.reward = valueOfReward
        self.getCost = getActionCost

        self.targetDictionary = targetDictionary
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation

        self.signalerInactionPossible = signalerInactionPossible


    def __call__(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = self.getActionUtility(targetAction, trueGoal, self.signalerLocation)
        speakerSignalPDF = self.getPragmaticSpeaker(trueGoal)
        #print(speakerSignalPDF)
        signalDF = pd.DataFrame(speakerSignalPDF.groupby(speakerSignalPDF.index.names).apply(lambda x: self.getExpectation(x, trueGoal)))
        #print(signalDF)
        signalDF.loc['do'+trueGoal] = dIYUtility

        if self.signalerInactionPossible:
            signalDF.loc['quit'] = 0
        print(signalDF)
        

        signalDF = signalDF.rename(columns={0: NC.PROBABILITY})
        signalDF[NC.PROBABILITY] = np.exp(self.rationality*signalDF[NC.PROBABILITY])
        signalDF[NC.PROBABILITY] = signalDF[NC.PROBABILITY]/sum(signalDF[NC.PROBABILITY])
        

        return(signalDF)

    def getExpectation(self, x2, trueGoal):
        signal = x2.index.get_level_values(NC.SIGNALS)[0]
        receiverPDF = self.getPragmaticReceiver(signal)
        print(x)
        print(receiverPDF)
        
        getIntentionLocation = lambda intention: [loc for loc, item in self.targetDictionary.items() if item == intention][0]
        getReceiverUtility = lambda x: self.getActionUtility(getIntentionLocation(x.index.get_level_values(NC.INTENTIONS)[0]), trueGoal, self.receiverLocation)*x[x.columns[0]]
        utilities = receiverPDF.groupby(receiverPDF.index.names).apply(getReceiverUtility)
        print(sum(utilities))
        return(sum(utilities))

    def getActionUtility(self, action, goal, startingPoint):
        utility = self.getCost(startingPoint, action)
        if self.targetDictionary[action] == goal:
            utility += self.reward
        return(utility)


#####################################################################################################################
# Decides between communicating and doing FIRST - not the accepted version
####################################################################################################################
class SpeakerActionSignalDistribution_SeparatedForCommunication():
    def __init__(self, getPragmaticSpeaker, getPragmaticReceiver, targetDictionary, signalerLocation, receiverLocation, rationality, valueOfReward, getActionCost):
        self.getPragmaticSpeaker = getPragmaticSpeaker
        self.getPragmaticReceiver = getPragmaticReceiver

        self.rationality = rationality
        self.reward = valueOfReward
        self.getCost = getActionCost

        self.targetDictionary = targetDictionary
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation


    def __call__(self, trueGoal):
        targetAction = [loc for loc, item in self.targetDictionary.items() if item == trueGoal][0]
        dIYUtility = self.getActionUtility(targetAction, trueGoal, self.signalerLocation)

        speakerSignalPDF = self.getPragmaticSpeaker(trueGoal)
        expectedUtilityOfCommunication  = self.getCommunicationUtility(speakerSignalPDF, trueGoal)
        pDIY, pSignal = self.normalizeUtilToPDF([dIYUtility, expectedUtilityOfCommunication])
        signalDF = speakerSignalPDF[speakerSignalPDF.columns[0]]*pSignal
        signalDF.loc[trueGoal] = pDIY
        signalDF = signalDF.to_frame()
        return(signalDF)

    def normalizeUtilToPDF(self, utilList):
        commPDF = [np.exp(self.rationality*util) for util in utilList]
        normCostant = sum(commPDF)
        normalizedCommPdf = [p/normCostant for p in commPDF]
        return(normalizedCommPdf)

    def getCommunicationUtility(self, speakerPDF, trueGoal):
        signalDF = pd.DataFrame(speakerPDF.groupby(speakerPDF.index.names).apply(lambda x: self.getExpectation(x, trueGoal)))
        commUtil = sum(signalDF.columns[0]*speakerPDF.columns[0])
        return(commUtil)

    def getExpectation(self, x, trueGoal):
        signal = x.index.get_level_values(NC.SIGNALS)[0]
        receiverPDF = self.getPragmaticReceiver(signal)
        getIntentionLocation = lambda intention: [loc for loc, item in self.targetDictionary.items() if item == intention][0]
        getReceiverUtility = lambda x: self.getActionUtility(getIntentionLocation(x.index.get_level_values(NC.INTENTIONS)[0]), trueGoal, self.receiverLocation)*x[x.columns[0]]
        utilities = receiverPDF.groupby(receiverPDF.index.names).apply(getReceiverUtility)
        return(sum(utilities))

    def getActionUtility(self, action, goal, startingPoint):
        utility = self.getCost(startingPoint, action)
        if self.targetDictionary[action] == goal:
            utility += self.reward
        return(utility)


