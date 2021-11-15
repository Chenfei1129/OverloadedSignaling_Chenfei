import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd
import numpy as np

import Algorithms.constantNames as NC


#helper functions for pandas
def getMultiIndexMindSpace(mindLevelsDictionary, columnNameList=None):
    conditions = list(mindLevelsDictionary.keys())
    levelValues = list(mindLevelsDictionary.values())
    speakerMindIndex = pd.MultiIndex.from_product(levelValues, names=conditions)
    jointMind = pd.DataFrame(index=speakerMindIndex, columns =columnNameList)
    return(jointMind)

def normalizeValuesPdSeries(pandasSeries):
    totalSum = sum(pandasSeries)
    probabilities = pandasSeries.groupby(pandasSeries.index.names).apply(lambda x: x/totalSum)
    return(probabilities)

class GenerateMind(object):
    def __init__(self, getWorldProbability, getDesireProbability, getGoalProbability, getActionProbability):
        self.getWorldProbability = getWorldProbability
        self.getDesireProbability = getDesireProbability
        self.getGoalProbability = getGoalProbability
        self.getActionProbability = getActionProbability
        
    def __call__(self, mindSpaceDictionary):
        jointMindSpace = getMultiIndexMindSpace(mindSpaceDictionary)

        #getMindForCondition = lambda condition: self.getConditionMind(condition, mindSpaceDictionary)
        mindProbabilitySeries = jointMindSpace.groupby(jointMindSpace.index.names).apply(lambda x: self.getConditionMind(x, mindSpaceDictionary))
        mindProbability = pd.DataFrame(normalizeValuesPdSeries(mindProbabilitySeries))
        mindProbability.rename(columns={0 : NC.P_MIND}, inplace=True)
        return(mindProbability)
    
    def getConditionMind(self, oneMindCondition, mindSpace):
        world = oneMindCondition.index.get_level_values(NC.WORLDS)[0]
        desire = oneMindCondition.index.get_level_values(NC.DESIRES)[0]
        goal = oneMindCondition.index.get_level_values(NC.INTENTIONS)[0]
        action = oneMindCondition.index.get_level_values(NC.ACTIONS)[0]
        
        worldSpace = mindSpace[NC.WORLDS]
        worldPDF = self.getWorldProbability(worldSpace)
        
        desireSpace = mindSpace[NC.DESIRES]
        desirePDF = self.getDesireProbability(desireSpace)
        
        goalSpace = mindSpace[NC.INTENTIONS]
        conditionalGoalPDF = self.getGoalProbability(goalSpace, world, desire)

        actionSpace = mindSpace[NC.ACTIONS]
        conditionalActionPDF = self.getActionProbability(actionSpace, world, goal)

        worldProb = worldPDF.loc[world].values[0]
        desireProb = desirePDF.loc[desire].values[0]
        goalProb = conditionalGoalPDF.loc[goal].values[0]

        # transforms in place index in order to handle nested tuples in pandas indexing
        actionProb = conditionalActionPDF.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(action)].values[0]

        mindProbability = worldProb*desireProb*goalProb*actionProb
        
        if type(mindProbability) != float:
            mindProbability = mindProbability[0]
        
        return(mindProbability) 