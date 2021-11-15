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
        #print(targetSpace)
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
        #put in a dictionary. {feature name: possible} loop through the disctionary. {feature name: thing I can talk about in the environment} itertools package combination. recursive function
        colors = [feature for feature in ['green', 'purple', 'red'] if feature in featureList]
        shapes = [feature for feature in ['circle', 'triangle', 'square' ] if feature in featureList]
        sizes = [feature for feature in ['small', 'medium', 'large'] if feature in featureList]
        shades = [feature for feature in ['noshaded', 'halfshaded', 'shaded' ] if feature in featureList]
        for color in colors :
            for shape in shapes:
                for size in sizes:
                    for shade in shades:
                        featureList.append(color + ' ' + shape+ ' ' + size + ' ' + shade)
                      
        for color in colors :
            for shape in shapes:
                for size in sizes:
                    featureList.append(color + ' ' + shape+ ' ' + size )                   
        for color in colors :
            for shape in shapes:
                for shade in shades:
                    featureList.append(color + ' ' + shape+ ' ' + shade )
        for color in colors :
            for size in sizes:
                for shade in shades:
                    featureList.append(color + ' ' + size+ ' ' + shade )
        for color in colors :
            for shape in shapes:
                for shade in shades:
                    featureList.append(color + ' ' + shape + ' ' + shade )      
                    #+ ' ' + size
        for color in colors :
            for shape in shapes :
                featureList.append(color + ' ' + shape)
        for color in colors :
            for size in sizes :
                featureList.append(color + ' ' + size)
        for shape in shapes :
            for size in sizes :
                featureList.append(shape + ' ' + size)
                
        for shape in shapes:
            for shade in shades:
                featureList.append(shape + ' ' + shade)
        for color in colors :
            for shade in shades :
                featureList.append(color + ' ' + shade)
        for size in sizes :
            for shade in shades :
                featureList.append(size + ' ' + shade)
 
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
    
import random
import itertools as it
class SampleExperimentEnvironment2():
    def __init__(self, fixedEnvironmentParameters, fluidParameterSpaces):
        self.fixedEnvironmentParameters = fixedEnvironmentParameters
        self.parameterSpaces = fluidParameterSpaces
    def __call__(self, numItem, numberOfSignals, vocabProportion=False):
        targetSpace = self.parameterSpaces['targets'].copy()
        target = random.choice(targetSpace)
        featureList = self.getFeaturesOfSampledTarget(target)
        itemBaseLine = self.getAllTargetsOfFeatures(targetSpace, target, featureList)
        
        possibleItems = []

        for items in itemBaseLine:
            item = random.choice(items)
            targetSpace.remove(item)
            possibleItems.append(item)
        possibleItems.append(target)
        for i in range(numItem - 4):
            item = random.choice(targetSpace)
            possibleItems.append(item)
        targetSpace = self.parameterSpaces['targets']
        targetDictionary = self.getTargetDictionary(possibleItems)
        featureSet = self.getFeaturesOfSampledTargets(possibleItems)
        if vocabProportion:
            #print(featureSet)
            signalSpace = self.sampleVocabUsingProportion(possibleItems, numberOfSignals, featureSet)
        else:

            numberOfSignals = min(numberOfSignals, len(featureSet)) #can't have more signals than in the feature set
            signalSpace = self.sampleUniformSignalSpace(numberOfSignals, featureSet)
            #print(signalSpace)
        propOfTargetSignals = self.getProportionTargetFeaturesInVocab(target,signalSpace)
        return(target, signalSpace, targetDictionary, numItem, propOfTargetSignals)
    
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
    
    def getFeaturesOfSampledTarget(self, target): # a string. Make the string a list of the features, combination 2 of 3.
        featureList = str.split(target)
        necessaryFeatureCombinations = [x for x in it.combinations(featureList, len(featureList)-1)]
        return necessaryFeatureCombinations

    def getAllTargetsOfFeatures(self, targetSpace, target, featureList):
        possibleItem = []
        item_space = targetSpace.remove(target)
        for feature in featureList:
            one_feature = []
            for item in targetSpace:
                if self.helper_ListContain(feature, item.split()):
                    one_feature.append(item)
            possibleItem.append(one_feature)
        return possibleItem
               
    def helper_ListContain(self, a, b):
        inOrNot = True
        for item in a:
            if item not in b:
                inOrNot = False
        return inOrNot
    def getFeaturesOfSampledTargets(self, targets):
        allFeatures = [tgt.split() for tgt in targets]
        #allFeatures.append(tgt.split()[1] for tgt in targets)
        #allFeatures.append(targets)#
        flatten = lambda l: [item for sublist in l for item in sublist]
        featureList = list(set(flatten(allFeatures)))
        #put in a dictionary. {feature name: possible} loop through the disctionary. {feature name: thing I can talk about in the environment} itertools package combination. recursive function
        colors = [feature for feature in ['green', 'purple', 'red', 'blue'] if feature in featureList]
        shapes = [feature for feature in ['circle', 'triangle', 'square' ] if feature in featureList]
        sizes = [feature for feature in ['small', 'medium', 'large' ] if feature in featureList]
        for color in colors :
            for shape in shapes:
                for size in sizes:
                    featureList.append(color + ' ' + shape+ ' ' + size )
                    #+ ' ' + size
        for color in colors :
            for shape in shapes :
                featureList.append(color + ' ' + shape)
        for color in colors :
            for size in sizes :
                featureList.append(color + ' ' + size)
        for shape in shapes :
            for size in sizes :
                featureList.append(shape + ' ' + size)

        return(featureList)
    def getProportionTargetFeaturesInVocab(self, trueIntention, signalSpace):
        nRelevantSignalsInVocab = sum([1 if feature in signalSpace else 0 for feature in trueIntention.split()])
        nFeatures = len(trueIntention.split())
        propRelevantSignals = nRelevantSignalsInVocab/nFeatures
        return(propRelevantSignals)
