import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))

import numpy as np
import itertools

import Algorithms.constantNames as NC

"""
    Constructed with:
    Fixed environment parameters dictionary must have: {'gridSize': (gridWidth, gridHeight), 
                                                        'signalerPosition': (sx, sy), 
                                                        'receiverPosition': (rx,ry)}
    Fluid parameter space must have: {'targets': [itemSpace],'signals': [signalSpace]}
    
    Called With:
    number of items in trial, number or proportion of signals, if vocab proportion  = True, signals read as a proportion
"""

class SampleExperimentEnvironment():
    def __init__(self, fixedEnvironmentParameters, fluidParameterSpaces):
        self.fixedEnvironmentParameters = fixedEnvironmentParameters
        self.parameterSpaces = fluidParameterSpaces
        
    def __call__(self, numberOfPossibleTargets, numberOfSignals, vocabProportion=False):
        
        targetSpace = self.sampleUniformTargetSpace(numberOfPossibleTargets)
        featureSet = self.getFeaturesOfSampledTargets(targetSpace)
        targetDictionary = self.getTargetDictionary(targetSpace)
        trueIntention = np.random.choice(targetSpace, 1)[0]
        if vocabProportion:
            #print(featureSet)
            signalSpace = self.sampleVocabUsingProportion(targetSpace, numberOfSignals, featureSet)
        else:

            numberOfSignals = min(numberOfSignals, len(featureSet)) #can't have more signals than in the feature set
            signalSpace = self.sampleUniformSignalSpace(numberOfSignals, featureSet)
            #print(signalSpace)
        propOfTargetSignals = self.getProportionTargetFeaturesInVocab(trueIntention,signalSpace)
        print(signalSpace)
        return(trueIntention, signalSpace, targetDictionary, numberOfPossibleTargets, propOfTargetSignals)

    #Sample the available vocabulary (by number or proportion)
    def sampleUniformSignalSpace(self, nSignals, signalsToSampleFrom = None):
        if not signalsToSampleFrom:
            signalsToSampleFrom = self.parameterSpaces[NC.SIGNALS]
        signalSpace  = np.random.choice(signalsToSampleFrom, nSignals, replace = False)
        return(list(signalSpace))

    def sampleVocabUsingProportion(self, targets, proportion, featureList):
        if isinstance(proportion, list):
            sampledProportion = np.random.uniform(low=min(proportion), high=max(proportion))
        else:
            sampledProportion = proportion
        #Proportion must be between 0 and 1
        sampledProportion = max(0, sampledProportion)
        sampledProportion = min(1, sampledProportion)
        # Must be at least one signal available
        numberOfSignals = max(round(sampledProportion*len(featureList)), 1)
        signalSpace = self.sampleUniformSignalSpace(numberOfSignals, featureList)#function outisde the class. Need function to tell what's your signal space. (Take sampling based or multiple words thing). 
        return(list(signalSpace))

    #Sample the items in the space
    def sampleUniformTargetSpace(self, nItemsInTrial):
        totalNItems = len(self.parameterSpaces['targets'])
        nItemsToSample = min(nItemsInTrial, totalNItems)
        sampledItemSpace  = np.random.choice(self.parameterSpaces['targets'], nItemsToSample, replace = False)
        return(sampledItemSpace)
    
    #Sample the location of the items
    def getTargetDictionary(self, targetSpace):
        gridWidth, gridHeight = self.fixedEnvironmentParameters['gridSize']
        availableSpaces = list(itertools.product(range(gridWidth), range(gridHeight)))
        availableSpaces.remove(self.fixedEnvironmentParameters['signalerPosition'])
        availableSpaces.remove(self.fixedEnvironmentParameters['receiverPosition'])

        nLocationsToSample =  len(targetSpace)
        sampleIndex = np.random.choice(np.arange(len(availableSpaces)), nLocationsToSample, replace = False)
        sampledPositions = [availableSpaces[indx] for indx in sampleIndex]
        targetDictionary = {location: target for location, target in zip(sampledPositions,targetSpace)}
        return(targetDictionary)

    def getFeaturesOfSampledTargets2(self, targets):
        allFeatures = [tgt.split() for tgt in targets]
        #allFeatures.append(tgt.split()[1] for tgt in targets)
        #allFeatures.append(targets)#
        flatten = lambda l: [item for sublist in l for item in sublist]
        featureList = list(set(flatten(allFeatures)))
        
        return(featureList)

    def getFeaturesOfSampledTargets(self, targets):
        allFeatures = [tgt.split() for tgt in targets]
        #allFeatures.append(tgt.split()[1] for tgt in targets)
        #allFeatures.append(targets)#
        flatten = lambda l: [item for sublist in l for item in sublist]
        featureList = list(set(flatten(allFeatures)))
        colors = [feature for feature in ['green', 'purple', 'red', 'blue'] if feature in featureList]
        shapes = [feature for feature in ['circle', 'triangle', 'square' ] if feature in featureList]
        sizes = [feature for feature in ['small', 'medium', 'large' ] if feature in featureList]
        for color in colors :
            for shape in shapes:
                #for size in sizes:
                featureList.append(color + ' ' + shape )#+ ' ' + size

        return(featureList)

    def getProportionTargetFeaturesInVocab(self, trueIntention, signalSpace):
        nRelevantSignalsInVocab = sum([1 if feature in signalSpace else 0 for feature in trueIntention.split()])
        nFeatures = len(trueIntention.split())
        propRelevantSignals = nRelevantSignalsInVocab/nFeatures
        return(propRelevantSignals)
    
    def getFeaturesOfSampledTargets2(self, targets):
        
        allFeatures = [tgt.split() for tgt in targets]
        allFeatures.append(targets)#
        flatten = lambda l: [item for sublist in l for item in sublist]
        featureList = list(set(flatten(allFeatures)))
        return(featureList)
    
    def getFeaturesOfSampledTargets2(self, targets):
        numFeature = len(targets[0].split())
        print(targets[0].split()[0])
        allFeatures = [tgt.split() for tgt in targets]
        #print(targets)
        allFeatures.append(targets)#
        flatten = lambda l: [item for sublist in l for item in sublist]
        featureList = list(set(flatten(allFeatures)))
        return(featureList)
