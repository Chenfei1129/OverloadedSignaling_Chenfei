import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd 
import numpy as np
import warnings

import Algorithms.constantNames as NC

def costOfMessage(message, dist = -2):
    return(-abs(dist))

def consistencyLexiconRSA_Misyak(utterance, targetState, meaning, openSignalKey = '1'):
    if meaning == openSignalKey:
        consistentSignals = [int(w == 1) for token, w in zip(utterance, targetState) if token == 1]  
    else:
        consistentSignals = [int(w != 1) for u, w in zip(utterance, targetState) if u == 1]
    return(consistentSignals.count(0) == 0)

def normalizeDictionary(unnormalizedDictionary):
    if sum(unnormalizedDictionary.values()) !=0:
        normalizationFactor = 1.0/sum(unnormalizedDictionary.values())
    else:
        warnings.warn('RSA: Normalizing a distribution w/o mass')
        normalizationFactor = 1.0/10E-10
    normalizedDictionary = {key : probability*normalizationFactor for key, probability in unnormalizedDictionary.items()}
    return(normalizedDictionary)

# RSA Speaker Models
class LiteralSpeaker():
    def __init__(self, lexicon, messageSet, messageCostFunction, lambdaRationalityParameter):
        self.lexicon = lexicon
        self.messageSet = messageSet
        self.messageCostFunction = messageCostFunction
        self.rationalityLambda = lambdaRationalityParameter
        
    def __call__(self, world, meaning):
        literalSpeakerUnnormalized = {message : self.getUtility(message, world, meaning) for message in self.messageSet}
        literalSpeakerPDF = normalizeDictionary(literalSpeakerUnnormalized)
        literalSpeakerDF = pd.DataFrame.from_dict(literalSpeakerPDF, orient = 'index').rename(columns={0: NC.LITERALSPEAKER})
        literalSpeakerDF.index.name = NC.SIGNALS    
        return(literalSpeakerDF)

    def getUtility(self, message, world, meaning, epsilon=10E-10):
        getLog = lambda x: np.log(x+epsilon) if x == 0 else np.log(x)
        utility = np.exp(self.rationalityLambda*(getLog(self.lexicon(message, world, meaning))-self.messageCostFunction(message)))
        return(utility)


class PragmaticListener():
    def __init__(self, getSpeaker, targetPrior, meaningPrior):
        self.targetPrior = targetPrior
        self.meaningPrior = meaningPrior
        self.getSpeaker = getSpeaker
        
    def __call__(self, message):
        pragmaticListenerUnnormalized = {world : pWorld *sum([self.getSignalerLikelihood(world, message, meaning)*pMeaning 
                                                for meaning, pMeaning in self.meaningPrior.items()])
                                                for world, pWorld in self.targetPrior.items()
                                                }
        pragmaticListenerPDF = normalizeDictionary(pragmaticListenerUnnormalized)
        pragmaticListenerDF = pd.DataFrame.from_dict(pragmaticListenerPDF, orient = 'index').rename(columns={0: NC.PRAGMATICLISTENER})
        pragmaticListenerDF.index.name = NC.INTENTIONS
        return(pragmaticListenerDF)

    def getSignalerLikelihood(self, world, message, meaning):
        signalPDF = self.getSpeaker(world, meaning) 
        messageProbability = signalPDF.loc[[message]].values[0]
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
    
    def getUtilityOfMessage(self, message, world):
        pdfOfTargetItems = self.getListener(message)
        targetProbability = pdfOfTargetItems.loc[[world]].values[0]
        epsilon = 10E-10
        getLog = lambda x: np.log(x+epsilon) if x == 0 else np.log(x)
        utilityOfMessage = np.exp(self.rationalityLambda*(getLog(targetProbability) - self.messageCostFunction(message)))
        return(utilityOfMessage)


# Listener RSA Model
class LiteralListener():
    def __init__(self, worldPriorDictionary, meaningPriorDictionary, lexiconFunction):
        self.worldPriors = worldPriorDictionary
        self.meaningPriors = meaningPriorDictionary
        self.lexicon = lexiconFunction
        
    def __call__(self, message):
        literalLisenerUnnormalized = {(state, meaning) : self.lexicon(message, state, meaning)*pState*pMeaning 
                                                for state, pState in self.worldPriors.items() 
                                                for meaning, pMeaning in self.meaningPriors.items()}
        literalListener = normalizeDictionary(literalLisenerUnnormalized)
        literalListenerDF = pd.DataFrame.from_dict(literalListener,orient = 'index').rename(columns={0: NC.LITERALLISTENER})
        literalListenerDF.index.name = NC.INTENTIONS
        return(literalListenerDF)

class PragmaticSpeaker_ListenerModel():
    def __init__(self, getListener, lexicon, messageSet, messageCostFunction, lambdaRationalityParameter, pragmaticSpeakerName = NC.PRAGMATICSPEAKER):
        self.getListener = getListener
        self.messageSet = messageSet
        self.messageCostFunction = messageCostFunction
        self.rationalityLambda = lambdaRationalityParameter
        self.pragmaticSpeakerName = pragmaticSpeakerName
        
    def __call__(self, world, meaning):
        messageUtilities = {message: self.getUtilityOfMessage(message, world, meaning) for message in self.messageSet}
        pragmaticSpeakerPDF = normalizeDictionary(messageUtilities)
        pragmaticSpeakerDF = pd.DataFrame.from_dict(pragmaticSpeakerPDF, orient = 'index').rename(columns={0: self.pragmaticSpeakerName})
        pragmaticSpeakerDF.index.name = NC.SIGNALS
        return(pragmaticSpeakerDF)
    
    def getUtilityOfMessage(self, message, world, meaning):
        pdfOfTargetItems = self.getListener(message)
        targetProbability = pdfOfTargetItems.loc[[(world, meaning)]].values[0]
        epsilon = 10E-10
        getLog = lambda x: np.log(x+epsilon) if x == 0 else np.log(x)
        utilityOfMessage = np.exp(self.rationalityLambda*(getLog(targetProbability) - self.messageCostFunction(message)))
        return(utilityOfMessage)

class PragmaticListener_ListenerModel():
    def __init__(self, getSpeaker, targetPrior, meaningPrior):
        self.targetPrior = targetPrior
        self.meaningPrior = meaningPrior
        self.getSpeaker = getSpeaker
        
    def __call__(self, message):
        pragmaticListenerUnnormalized = {world : pWorld *sum([self.getSignalerLikelihood(world, message, meaning)*pMeaning 
                                                for meaning, pMeaning in self.meaningPrior.items()])
                                                for world, pWorld in self.targetPrior.items()
                                                }
        pragmaticListenerPDF = normalizeDictionary(pragmaticListenerUnnormalized)
        pragmaticListenerDF = pd.DataFrame.from_dict(pragmaticListenerPDF, orient = 'index').rename(columns={0: NC.PRAGMATICLISTENER})
        pragmaticListenerDF.index.name = NC.INTENTIONS
        return(pragmaticListenerDF)

    def getSignalerLikelihood(self, world, message, meaning):
        signalPDF = self.getSpeaker(world, meaning) 
        messageProbability = signalPDF.loc[[message]].values[0]
        return(messageProbability)