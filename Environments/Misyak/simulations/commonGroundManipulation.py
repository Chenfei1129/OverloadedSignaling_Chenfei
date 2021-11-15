import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import pandas as pd
import warnings
import random

import Algorithms.constantNames as NC
import Simulations.modelLabels as ML

from Environments.Misyak.simulations.setupInference_Misyak import SetupMisyakTrial
from Environments.Misyak.misyakConstruction import getSignalSpace, getActionSpace


class SimulateMisyakModelFromDF():
    def __init__(self, modelParameters):
        self.iterCounter = 0
        self.modelParameters = modelParameters

    def __call__(self, dfRow):
        setupTrial = SetupMisyakTrial(rationalityParameter= self.modelParameters['alpha'], 
            valueOfReward = self.modelParameters['valueOfReward'], 
            signalMeaningPrior = self.formatSignalPrior(self.modelParameters['signalMeaningPrior']),
            silencePossible = True, 
            signalCost = self.modelParameters['costOfSignal'], 
            costOfPunishment = self.modelParameters['costOfPunishment'],
            nBoxes = self.modelParameters['nBoxes'])
        nSignalsIW, nAxesIW, shadowIW, nRewardsIW = self.getCondition(dfRow)
        _, getReceiver, getSignaler, _ = setupTrial(nSignalsIW, nAxesIW, shadowIW, nRewardsIW)
        
        self.iterCounter += 1
        print('Run: ', self.iterCounter)

        world = dfRow['world']
        signalerTokenPDF = getSignaler({'worlds': world})
        signalerChoice = self.sampleSignalerChoice(signalerTokenPDF, dfRow['n_tokens'])[0]
        print("signaler choice: ", signalerChoice, "\n")
        receieverActionPDF = getReceiver(signalerChoice)
        receiverActionChoice = self.sampleReceiverChoice(receieverActionPDF, dfRow['n_axes'])

        print("Receiver choice: ", receiverActionChoice, "\n")
        rewardsAchieved = self.getRewardsAchieved(world, receiverActionChoice)

        newColumnNames = ['signalerChoice', 'receiverChoice', 'modelRewardsAchieved']
        output = pd.Series([signalerChoice, receiverActionChoice, rewardsAchieved], index=newColumnNames)
        return(output)

    def getCondition(self, dfRow, maxAxes=2, maxTokens=2):
        axesInCommonGround = dfRow['axesInCG']
        tokensInCommonGround = dfRow['tokensInCG']
        if axesInCommonGround:
            nAxes = dfRow['n_axes']
        else:
            nAxes = maxAxes
        if tokensInCommonGround:
            nTokens = dfRow['n_tokens']
        else:
            nTokens = maxTokens
        showShadow = dfRow['show_imprints']
        nRewards = dfRow['n_bananas']
        return(nTokens, nAxes, showShadow, nRewards)

    def sampleSignalerChoice(self, sigPDF, trueNumberOfTokens):
        signalSpace = getSignalSpace(nBoxes=self.modelParameters['nBoxes'], nSignals = trueNumberOfTokens)
        restrictedSignalPDF = self.restrictPDF(sigPDF, signalSpace)
        signal = restrictedSignalPDF.sample(weights = restrictedSignalPDF.columns[0]).index[0]
        return(signal)

    def sampleReceiverChoice(self, recPDF, nAxes):
        actionSpace = getActionSpace(nBoxes = self.modelParameters['nBoxes'], nReceiverChoices = nAxes)
        actionOnlyPDF = pd.DataFrame(recPDF.groupby(level=[NC.ACTIONS]).sum())
        restrictedActionPDF = self.restrictPDF(actionOnlyPDF, actionSpace)

        print(sum(restrictedActionPDF[restrictedActionPDF.columns[0]]), "\n")

        if round(sum(restrictedActionPDF[restrictedActionPDF.columns[0]]),8) != 1.0:
          warnings.warn("signaler signal not consistent with anything, receiver samples randomly.")
          receiverChoice = random.sample(actionSpace, k=1)
          receiverAction = receiverChoice[0] 
        else:  
            receiverChoice = restrictedActionPDF.sample(weights = restrictedActionPDF.columns[0])
            receiverAction = receiverChoice.index[0] #receiverChoice[list(recPDF.index.names).index('actions')]
        return(receiverAction)

    def restrictPDF(self, pdfDF, listOfAcceptableIndices):
        idxName = pdfDF.index.names
        isIndexAcceptable = lambda x: 1 if x.index.get_level_values(idxName[0])[0] in listOfAcceptableIndices else 0
        pdfDF['s'] = pdfDF.groupby(idxName).transform(isIndexAcceptable)
        
        colName = pdfDF.columns[0]
        pdfDF[colName] = pdfDF[colName]*pdfDF['s']
        pdfDF[colName] = pdfDF[colName]/pdfDF[colName].sum()
        pdfDF = pdfDF.drop('s', axis=1)
        return(pdfDF)

    def getRewardsAchieved(self, world, receiverAction):
        bananasAchieved = [box+ax for box, ax in zip(world, receiverAction)]
        return(bananasAchieved.count(2))

    def formatSignalPrior(self, probability, categoryNames = ['1', '-1'], decimalPlaces = 7):
        prior = {categoryNames[0]:round(probability, decimalPlaces), categoryNames[1]: round(1-probability, decimalPlaces)}
        return(prior)
