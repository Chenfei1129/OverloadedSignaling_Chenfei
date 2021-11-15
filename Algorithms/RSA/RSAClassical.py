import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd 
import numpy as np
import warnings

import Algorithms.constantNames as NC

def costOfMessage(message, dist = 0):
    return(-abs(dist))

def calculateLocationCost_TaxicabMetric(agentLocation, proposedActionLocation):
    getDistance = lambda a, b: abs(a-b)
    absoluteActionCosts = [getDistance(locationXY, actionXY) for locationXY, actionXY in zip(agentLocation, proposedActionLocation)]
    totalActionCost = -sum(absoluteActionCosts)
    return(totalActionCost)

def consistencyLexiconRSA_Experiment(utterance, targetState):
    utteranceFeatures = list(utterance.split())
    featuresOfTarget = list(targetState.split())
    consistentUtterance = [feature in featuresOfTarget for feature in utteranceFeatures]
    return(all(consistentUtterance))

def normalizeDictionary(unnormalizedDictionary):
    #print(unnormalizedDictionary)
    if sum(unnormalizedDictionary.values()) !=0:
        normalizationFactor = 1.0/sum(unnormalizedDictionary.values())
    else:
        #print("*****")
        warnings.warn('RSA: Normalizing a distribution w/o mass')
        normalizationFactor = 1.0/10E-10
    normalizedDictionary = {key : probability*normalizationFactor for key, probability in unnormalizedDictionary.items()}
    return(normalizedDictionary)

class PragmaticListener():
    def __init__(self, getSpeaker, targetPrior):
        self.targetPrior = targetPrior
        self.getSpeaker = getSpeaker
        
    def __call__(self, message):
        pragmaticListenerUnnormalized = {world : self.getSignalerLikelihood(world, message)*prior for world, prior in self.targetPrior.items()}
        pragmaticListenerPDF = normalizeDictionary(pragmaticListenerUnnormalized)
        pragmaticListenerDF = pd.DataFrame.from_dict(pragmaticListenerPDF, orient = 'index').rename(columns={0: NC.PRAGMATICLISTENER})
        pragmaticListenerDF.index.name = NC.INTENTIONS
        return(pragmaticListenerDF)

    def getSignalerLikelihood(self, world, message):
        signalPDF = self.getSpeaker(world) 
        messageProbability = signalPDF.loc[message].values[0]
        return(messageProbability)

class PragmaticSpeaker():
    def __init__(self, getListener, messageSet, messageCostFunction, lambdaRationalityParameter, pragmaticSpeakerName = NC.PRAGMATICSPEAKER):
        self.getListener = getListener
        self.messageSet = messageSet
        self.messageCostFunction = messageCostFunction
        self.rationalityLambda = lambdaRationalityParameter
        self.pragmaticSpeakerName = pragmaticSpeakerName
        
    def __call__(self, world, utilities = False):
        messageUtilities = {message: self.getUtilityOfMessage(message, world) for message in self.messageSet}
        if utilities:
            pragmaticSpeakerDF = pd.DataFrame.from_dict(messageUtilities, orient = 'index').rename(columns={0: NC.UTILITY})

        else:
            pragmaticSpeakerPDF = normalizeDictionary(messageUtilities)
            pragmaticSpeakerDF = pd.DataFrame.from_dict(pragmaticSpeakerPDF, orient = 'index').rename(columns={0: self.pragmaticSpeakerName})
        pragmaticSpeakerDF.index.name = NC.SIGNALS
        return(pragmaticSpeakerDF)
    
    def getUtilityOfMessage(self, message, world, epsilon = 10E-10):
        pdfOfTargetItems = self.getListener(message)
        
        targetProbability = pdfOfTargetItems.loc[world].values[0]
        
        getLog = lambda x: np.log(x+epsilon) if x == 0 else np.log(x)
        utilityOfMessage = np.exp(self.rationalityLambda*(getLog(targetProbability) - self.messageCostFunction(message)))
        #print(utilityOfMessage)
        
        return(utilityOfMessage)

class LiteralListener():
    def __init__(self, worldPriorDictionary, lexiconFunction):
        self.worldPriors = worldPriorDictionary
        self.lexicon = lexiconFunction
        
    def __call__(self, message):
        literalLisenerUnnormalized = {state : self.lexicon(message, state)*probability for state, probability in self.worldPriors.items()}
        literalListener = normalizeDictionary(literalLisenerUnnormalized)
        literalListenerDF = pd.DataFrame.from_dict(literalListener,orient = 'index').rename(columns={0: NC.LITERALLISTENER})
        literalListenerDF.index.name = NC.INTENTIONS
        return(literalListenerDF)

class LiteralSpeaker():
    def __init__(self, lexicon, messageSet, messageCostFunction, lambdaRationalityParameter):
        self.lexicon = lexicon
        self.messageSet = messageSet
        self.messageCostFunction = messageCostFunction
        self.rationalityLambda = lambdaRationalityParameter
        
    def __call__(self, world):
        literalSpeakerUnnormalized = {message : self.getUtility(message, world) for message in self.messageSet}
        literalSpeakerPDF = normalizeDictionary(literalSpeakerUnnormalized)
        literalSpeakerDF = pd.DataFrame.from_dict(literalSpeakerPDF, orient = 'index').rename(columns={0: NC.LITERALSPEAKER})
        literalSpeakerDF.index.name = NC.SIGNALS    
        return(literalSpeakerDF)

    def getUtility(self, message, world, epsilon=10E-10):
        getLog = lambda x: np.log(x+epsilon) if x == 0 else np.log(x)
        utility = np.exp(self.rationalityLambda*(getLog(self.lexicon(message, world))-self.messageCostFunction(message)))
        return(utility)
