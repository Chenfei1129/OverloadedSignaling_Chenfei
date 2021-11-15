import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd 
import numpy as np

import Algorithms.constantNames as NC

from Algorithms.RSA.RSAClassical import *
from Algorithms.RSA.RSAExtensionsForActionMoreWord import SpeakerActionSignalDistribution

class SetupExperiment_RSAListenerInference():
    def __init__(self, rationality, getCost = lambda x: 0, lexicon = consistencyLexiconRSA_Experiment):
        self.rationality = rationality
        self.getCost = getCost
        self.lexicon = lexicon

    def __call__(self, targetPrior, signalSpace, maxLayer = 1, getAllLayers = False):
        currentLayer = 0
        getListener = LiteralListener(worldPriorDictionary=targetPrior, lexiconFunction=self.lexicon)
        listenerDict = {str(currentLayer):getListener}

        while currentLayer < maxLayer:
            getPragmaticSpeaker = PragmaticSpeaker(getListener=getListener, 
                                           messageSet=signalSpace, 
                                           messageCostFunction=self.getCost, 
                                           lambdaRationalityParameter=self.rationality)
            getListener  = PragmaticListener(getSpeaker = getPragmaticSpeaker, targetPrior = targetPrior)
            currentLayer += 1
            listenerDict[str(currentLayer)] = getListener

        if getAllLayers:
            return(listenerDict)
        else:
            return(listenerDict[str(currentLayer)])

class SetupExperiment_RSASpeakerInference():
    def __init__(self, rationality, getCost = lambda x:0, lexicon = consistencyLexiconRSA_Experiment):
        self.rationality = rationality
        self.getCost = getCost
        self.lexicon = lexicon

    def __call__(self, targetPrior, signalSpace, maxLayer = 1, getAllLayers = False):
        currentLayer = 0
        getSpeaker = LiteralSpeaker(lexicon = self.lexicon, 
            messageSet=signalSpace, 
            messageCostFunction=self.getCost, 
            lambdaRationalityParameter=self.rationality)
        speakerDict = {str(currentLayer): getSpeaker}

        while currentLayer < maxLayer:
            getPragmaticListener  = PragmaticListener(getSpeaker=getSpeaker, targetPrior=targetPrior)
            getSpeaker = PragmaticSpeaker(getListener=getPragmaticListener, 
                                           messageSet=signalSpace, 
                                           messageCostFunction=self.getCost, 
                                           lambdaRationalityParameter=self.rationality)
            currentLayer += 1
            speakerDict[str(currentLayer)] = getSpeaker

        if getAllLayers:
            return(speakerDict)
        else:
            return(speakerDict[str(currentLayer)])


class SetupExperiment_RSASpeakerWithActionChoice():
    def __init__(self, rationality, valueOfReward, SpeakerWActionPDF = SpeakerActionSignalDistribution, getActionCost = calculateLocationCost_TaxicabMetric, getSignalingCost = costOfMessage, lexicon = consistencyLexiconRSA_Experiment, signalerInactionPossible=False):
        self.rationality = rationality
        self.valueOfReward = valueOfReward
        self.getActionCost = getActionCost
        self.getCost = getSignalingCost
        self.lexicon = lexicon
        self.SpeakerWActionPDF = SpeakerWActionPDF
        self.signalerInactionPossible = signalerInactionPossible
        

    def __call__(self, targetPrior, signalSpace, targetDictionary, signalerLocation, receiverLocation, maxLayer = 1, getAllLayers = False):
        currentLayer = 0
        #getListener = LiteralListener(worldPriorDictionary=targetPrior, lexiconFunction=self.lexicon)
        getSpeaker = LiteralSpeaker(lexicon = self.lexicon, 
                                    messageSet=signalSpace, 
                                    messageCostFunction=self.getCost, 
                                    lambdaRationalityParameter=self.rationality)

        #getActingSpeaker = self.SpeakerWActionPDF(getPragmaticSpeaker=getSpeaker, getPragmaticReceiver=getListener, targetDictionary=targetDictionary, signalerLocation=signalerLocation, receiverLocation=receiverLocation, rationality=self.rationality, valueOfReward = self.valueOfReward, getActionCost=self.getActionCost)
        speakerDict = {str(currentLayer):getSpeaker} #RESTRICTS SPEAKER 0 TO ONLY SEND A SIGNAL -- THERE IS NO UTILITY MODEL OF OTHER AGENT, SO HOW TO COMPARE??

        while currentLayer < maxLayer:
            getPragmaticListener  = PragmaticListener(getSpeaker=getSpeaker, targetPrior=targetPrior)
            getSpeaker = PragmaticSpeaker(getListener=getPragmaticListener, 
                                           messageSet=signalSpace, 
                                           messageCostFunction=self.getCost, 
                                           lambdaRationalityParameter=self.rationality)

            getActingSpeaker = self.SpeakerWActionPDF(getPragmaticSpeaker=getSpeaker, 
                                                    getPragmaticReceiver=getPragmaticListener, 
                                                    targetDictionary=targetDictionary, 
                                                    signalerLocation=signalerLocation, 
                                                    receiverLocation=receiverLocation, 
                                                    rationality=self.rationality, 
                                                    valueOfReward = self.valueOfReward, 
                                                    getActionCost=self.getActionCost, getSignalCost = self.getCost, 
                                                    signalerInactionPossible = self.signalerInactionPossible)
            currentLayer += 1
            speakerDict[str(currentLayer)] = getActingSpeaker

        if getAllLayers:
            return(speakerDict)
        else:
            return(speakerDict[str(currentLayer)])
            
