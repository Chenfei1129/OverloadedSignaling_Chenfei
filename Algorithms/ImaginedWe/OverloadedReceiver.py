import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd 

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace


"""
    commongroundDictionary: dictionary {'mind component label': list of items in mind component space}
    constructMind: function generating prior over possible target minds from a dictionary with mind component labels as keys and desired pandas indices as values,
    getSignalLikelihood: generative signaler function
    signaCategoryPrior: dictionary {signalType: probability of signalType}

class ReceiverZero(object):
    def __init__(self, commonGroundDictionary, constructMind, getSignalerZero, signalCategoryPrior = None):
        self.mindPrior = constructMind(commonGroundDictionary)
        self.getSignalLikelihood = getSignalerZero
        self.signalCategoryPrior = signalCategoryPrior

        #index and column names for dataframe
        self.mindLabels = list(commonGroundDictionary.keys())
        

    def __call__(self, signal):
        if self.signalCategoryPrior = None:
            mindAndCategoryPrior = self.mindPrior
        else:
            mindAndCategoryPrior = self.constructJointMindSignalCategoryPrior(self.mindPrior, self.signalCategoryPrior)
        likelihoodDF = self.constructLikelihoodDataFrameFromMindConditions(self.mindPrior)
        mindPosterior = self.getMindPosterior(mindAndCategoryPrior, likelihoodDF, signal)
        return(mindPosterior)

    def constructLikelihoodDataFrameFromMindConditions(self, mindPrior):
        categoryNames = list(self.signalCategoryPrior.keys())

        # find the signal likelihood distribution for each signaler type and concatenate dataframes into a single pandas DF distribution
        likelihoodByCategory = [self.getSignalLikelihood(mindPrior, signalerType) for signalerType in categoryNames]
        likelihoodDistributionList =  [pd.concat([likelihoodDist], keys=[categoryName], names=[NC.SINGALER_C]) for likelihoodDist, categoryName in zip(likelihoodByCategory, categoryNames)]
        likelihoodDistributionDF = pd.concat(likelihoodDistributionList)

        return(likelihoodDistributionDF)

    def constructJointMindSignalCategoryPrior(self, mindPrior, categoryPrior):
        #from signal category prior, create a pandas df with index as category type label and column of p(c) probability
        categoryPriorDF = pd.DataFrame(list(categoryPrior.items()), columns=[NC.SINGALER_C, NC.P_C])
        categoryPriorDF.set_index(NC.SINGALER_C, inplace=True)

        #duplicate the mind prior * the number of possible signal type categories, set the index to the joint p(mind, c) combinations
        categoryNames = list(categoryPrior.keys())
        numberOfCategories = len(categoryNames)
        mindCPrior = pd.concat([mindPrior]*numberOfCategories, keys=categoryNames, names=[NC.SINGALER_C])

        #merge the categoryPriorDF into the mindCPrior, take the product of p(mind)*p(c) columns and return the resulting column p(mind, c)
        jointPrior = pd.merge(left=mindCPrior.reset_index(level=self.mindLabels),right=categoryPriorDF,on=[NC.SINGALER_C])
        jointPrior[NC.P_JOINTPRIOR] = jointPrior[NC.P_MIND] * jointPrior[NC.P_C]
        jointPrior = jointPrior.set_index(self.mindLabels,append=True)[[NC.P_JOINTPRIOR]]
        return(jointPrior)

    def getMindPosterior(self, jointPrior, likelihood, signal):
        #merge the prior and likelihood dataframes, take the product of p(mind,c)*p(signal|mind,c) and get the posterior distribution 
        posterior = pd.merge(left=jointPrior,right=likelihood.reset_index(level=[NC.SIGNALS]),on=[NC.SINGALER_C]+self.mindLabels)
        posterior[NC.P_MINDPOSTERIOR] = posterior[NC.P_JOINTPRIOR] * posterior[NC.P_SIGNALLIKELIHOOD]
        posterior = posterior.set_index(posterior[NC.SIGNALS],append=True)[[NC.P_MINDPOSTERIOR]]
        posterior = posterior.reorder_levels([NC.SIGNALS,NC.SINGALER_C]+self.mindLabels)

        #extract the signal location, normalize, and integrate out the category type to get p(mind|signal)
        mindAndCPosterior = posterior.loc[signal]
        mindAndCPosterior[NC.P_MINDPOSTERIOR] = mindAndCPosterior[NC.P_MINDPOSTERIOR]/sum(mindAndCPosterior[NC.P_MINDPOSTERIOR])
        #print("posterior before integration", mindAndCPosterior, '\n')
        mindPosterior = mindAndCPosterior.groupby(level=self.mindLabels).sum()
        return(mindPosterior)
"""


"""
    commongroundDictionary: dictionary {'mind component label': list of items in mind component space}
    constructMind: function generating prior over possible target minds from a dictionary with mind component labels as keys and desired pandas indices as values,
    getSignalLikelihood: generative signaler function
    signaCategoryPrior: dictionary {signalType: probability of signalType}
"""
class ReceiverZero(object):
    def __init__(self, commonGroundDictionary, constructMind, getSignalerZero, signalCategoryPrior):
        self.mindPrior = constructMind(commonGroundDictionary)
        self.getSignalLikelihood = getSignalerZero
        self.signalCategoryPrior = signalCategoryPrior
        #index and column names for dataframe
        self.mindLabels = list(commonGroundDictionary.keys())
        self.posteriorTable = self.setPosteriorTable(self.mindPrior, signalCategoryPrior, getSignalerZero)
        

    def __call__(self, signal):
        mindAndCPosterior = self.posteriorTable.loc[signal]
        mindAndCPosterior[NC.P_MINDPOSTERIOR] = mindAndCPosterior[NC.P_MINDPOSTERIOR]/sum(mindAndCPosterior[NC.P_MINDPOSTERIOR])
        #print("posterior before integration", mindAndCPosterior, '\n')
        mindPosterior = mindAndCPosterior.groupby(level=self.mindLabels).sum()

        return(mindPosterior)

    def setPosteriorTable(self, mindPrior, signalCategoryPrior, getSignalLikelihood):
        """if signalCategoryPrior == None:
                                    mindAndCategoryPrior = mindPrior
                                else:"""
        mindAndCategoryPrior = self.constructJointMindSignalCategoryPrior(mindPrior, signalCategoryPrior)
        likelihoodDF = self.constructLikelihoodDataFrameFromMindConditions(mindPrior)

        #merge the prior and likelihood dataframes, take the product of p(mind,c)*p(signal|mind,c) and get the posterior distribution 
        posterior = pd.merge(left=mindAndCategoryPrior,right=likelihoodDF.reset_index(level=[NC.SIGNALS]),on=[NC.SINGALER_C]+self.mindLabels)
        posterior[NC.P_MINDPOSTERIOR] = posterior[NC.P_JOINTPRIOR] * posterior[NC.P_SIGNALLIKELIHOOD]
        posterior = posterior.set_index(posterior[NC.SIGNALS],append=True)[[NC.P_MINDPOSTERIOR]]
        posterior = posterior.reorder_levels([NC.SIGNALS,NC.SINGALER_C]+self.mindLabels)
        return(posterior)

    def constructLikelihoodDataFrameFromMindConditions(self, mindPrior):
        categoryNames = list(self.signalCategoryPrior.keys())

        # find the signal likelihood distribution for each signaler type and concatenate dataframes into a single pandas DF distribution
        likelihoodByCategory = [self.getSignalLikelihood(mindPrior, signalerType) for signalerType in categoryNames]
        likelihoodDistributionList =  [pd.concat([likelihoodDist], keys=[categoryName], names=[NC.SINGALER_C]) for likelihoodDist, categoryName in zip(likelihoodByCategory, categoryNames)]
        likelihoodDistributionDF = pd.concat(likelihoodDistributionList)

        return(likelihoodDistributionDF)

    def constructJointMindSignalCategoryPrior(self, mindPrior, categoryPrior):
        #from signal category prior, create a pandas df with index as category type label and column of p(c) probability
        categoryPriorDF = pd.DataFrame(list(categoryPrior.items()), columns=[NC.SINGALER_C, NC.P_C])
        categoryPriorDF.set_index(NC.SINGALER_C, inplace=True)

        #duplicate the mind prior * the number of possible signal type categories, set the index to the joint p(mind, c) combinations
        categoryNames = list(categoryPrior.keys())
        numberOfCategories = len(categoryNames)
        mindCPrior = pd.concat([mindPrior]*numberOfCategories, keys=categoryNames, names=[NC.SINGALER_C])

        #merge the categoryPriorDF into the mindCPrior, take the product of p(mind)*p(c) columns and return the resulting column p(mind, c)
        jointPrior = pd.merge(left=mindCPrior.reset_index(level=self.mindLabels),right=categoryPriorDF,on=[NC.SINGALER_C])
        jointPrior[NC.P_JOINTPRIOR] = jointPrior[NC.P_MIND] * jointPrior[NC.P_C]
        jointPrior = jointPrior.set_index(self.mindLabels,append=True)[[NC.P_JOINTPRIOR]]
        return(jointPrior)



    
