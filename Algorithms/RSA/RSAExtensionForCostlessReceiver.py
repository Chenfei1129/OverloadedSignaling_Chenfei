import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd 
import numpy as np

import Algorithms.constantNames as NC
from Algorithms.RSA.RSAClassical import *


class SpeakerActionSignalDistribution_NoReceiverCost():
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

        signalDF = pd.DataFrame(speakerSignalPDF.groupby(speakerSignalPDF.index.names).apply(lambda x: self.getExpectation(x, trueGoal)))
        signalDF.loc[trueGoal] = dIYUtility
        signalDF = signalDF.rename(columns={0: NC.PROBABILITY})
        signalDF[NC.PROBABILITY] = np.exp(self.rationality*signalDF[NC.PROBABILITY])
        signalDF[NC.PROBABILITY] = signalDF[NC.PROBABILITY]/sum(signalDF[NC.PROBABILITY])
        return(signalDF)

    def getExpectation(self, x, trueGoal):
        signal = x.index.get_level_values(NC.SIGNALS)[0]
        receiverPDF = self.getPragmaticReceiver(signal)
        getIntentionLocation = lambda intention: [loc for loc, item in self.targetDictionary.items() if item == intention][0]
        getReceiverUtility = lambda x: self.getActionUtility(getIntentionLocation(x.index.get_level_values(NC.INTENTIONS)[0]), trueGoal, self.receiverLocation, True)*x[NC.PRAGMATICLISTENER]
        utilities = receiverPDF.groupby(receiverPDF.index.names).apply(getReceiverUtility)
        return(sum(utilities))

    def getActionUtility(self, action, goal, startingPoint, receiver=False):
        utility = 0.0
        if not receiver:
            utility = self.getCost(startingPoint, action)
        if self.targetDictionary[action] == goal:
            utility += self.reward
        return(utility)

class SetupExperiment_RSASpeakerWithActionChoice_NoReceiverCost():
    def __init__(self, rationality, valueOfReward, getActionCost=calculateLocationCost_TaxicabMetric, getSignalingCost = lambda x: 0, lexicon = consistencyLexiconRSA_Experiment):
        self.rationality = rationality
        self.valueOfReward = valueOfReward

        self.getActionCost = getActionCost
        self.getCost = getSignalingCost
        self.lexicon = lexicon

    def __call__(self, targetPrior, signalSpace, targetDictionary, signalerLocation, receiverLocation, maxLayer = 1,getAllLayers = False):
        currentLayer = 0

        getSpeaker = LiteralSpeaker(lexicon = self.lexicon, messageSet=signalSpace, messageCostFunction=self.getCost, lambdaRationalityParameter=self.rationality)
        speakerDict = {str(currentLayer):getSpeaker}
        while currentLayer < maxLayer:
            getPragmaticListener  = PragmaticListener(getSpeaker=getSpeaker, targetPrior=targetPrior)
            getSpeaker = PragmaticSpeaker(getListener=getPragmaticListener, 
                                           messageSet=signalSpace, 
                                           messageCostFunction=self.getCost, 
                                           lambdaRationalityParameter=self.rationality)

            getActingSpeaker = SpeakerActionSignalDistribution_NoReceiverCost(getPragmaticSpeaker=getSpeaker, 
                                                    getPragmaticReceiver=getPragmaticListener, 
                                                    targetDictionary=targetDictionary, 
                                                    signalerLocation=signalerLocation, 
                                                    receiverLocation=receiverLocation, 
                                                    rationality=self.rationality, 
                                                    valueOfReward = self.valueOfReward, 
                                                    getActionCost=self.getActionCost)
            currentLayer += 1
            speakerDict[str(currentLayer)] = getActingSpeaker

        if getAllLayers:
            return(speakerDict)
        else:
            return(speakerDict[str(currentLayer)])