import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import pandas as pd
import numpy as np
import warnings
import random
import itertools

import Algorithms.constantNames as NC
import Simulations.modelLabels as ML

from Environments.Misyak.simulations.setupInference_Misyak import SetupMisyakTrial
from Environments.Misyak.simulations.setupRSAInference_Misyak import SetupRSAMisyakTrial, SetupRSAMisyakTrial_Receiver
from Environments.Misyak.misyakConstruction import getSignalSpace, getActionSpace

class SimulateMisyakModelFromDF():
    def __init__(self, modelParameters):
        self.iterCounter = 0
        self.modelParameters = modelParameters

    def __call__(self, dfRow):
        self.iterCounter += 1
        print('Run: ', self.iterCounter)
        world = dfRow['world']
        #print('world', world)

        getRSA_R0, getRSA_S1, getRSA_R1 = self.setupRSAInference(dfRow)
        # RSA S1R0
        s1r0RSA_signalerChoice = self.sampleSignalerChoice(getRSA_S1, world, iw=False) 
        sampledChoice_rsaR0 = self.sampleReceiverChoice(getRSA_R0, s1r0RSA_signalerChoice, iw=False)[0] #here have joint meaning and world inference
        s1r0RSA_receiverChoice = self.sampleRSAReceiverActionFromInference(sampledChoice_rsaR0, dfRow['n_axes'])
        s1r0RSA_rewardsAchieved = self.getRewardsAchieved(world, s1r0RSA_receiverChoice)
        #print('rsa s1r0 \n signaler choice: ', s1r0RSA_signalerChoice, 'receiver choice: ', s1r0RSA_receiverChoice, 'rewards ', s1r0RSA_rewardsAchieved)
                        
        # RSA S1R1
        s1r1RSA_signalerChoice = self.sampleSignalerChoice(getRSA_S1, world, iw=False)
        sampledChoice_rsaR1 = self.sampleReceiverChoice(getRSA_R1, s1r1RSA_signalerChoice, iw=False)
        s1r1RSA_receiverChoice = self.sampleRSAReceiverActionFromInference(sampledChoice_rsaR1, dfRow['n_axes'])
        s1r1RSA_rewardsAchieved = self.getRewardsAchieved(world, s1r1RSA_receiverChoice)
        #print('rsa s1r1 \n signaler choice: ', s1r1RSA_signalerChoice, 'receiver choice: ', s1r1RSA_receiverChoice, 'rewards ', s1r1RSA_rewardsAchieved)

        getIW_R0, getIW_S1, getIW_R1 = self.setupIWInference(dfRow)
        # IW S1R0
        s1r0_signalerChoice = self.sampleSignalerChoice(getIW_S1, world)
        s1r0_receiverChoice = self.sampleReceiverChoice(getIW_R0, s1r0_signalerChoice)
        s1r0_rewardsAchieved = self.getRewardsAchieved(world, s1r0_receiverChoice)
        #print('iw s1r0 \n signaler choice: ', s1r0_signalerChoice, 'receiver choice: ', s1r0_receiverChoice, 'rewards ', s1r0_rewardsAchieved)

        # IW S1R1
        s1r1_signalerChoice = self.sampleSignalerChoice(getIW_S1, world)
        s1r1_receiverChoice = self.sampleReceiverChoice(getIW_R1, s1r1_signalerChoice)
        s1r1_rewardsAchieved = self.getRewardsAchieved(world, s1r1_receiverChoice)
        #print('iw s1r1 \n signaler choice: ', s1r1_signalerChoice, 'receiver choice: ', s1r1_receiverChoice, 'rewards ', s1r1_rewardsAchieved)


        newColumnNames = ['IW_S1R0_signalerChoice', 'IW_S1R0_receiverChoice', 'IW_S1R0_modelRewardsAchieved', 
                            'IW_S1R1_signalerChoice', 'IW_S1R1_receiverChoice', 'IW_S1R1_modelRewardsAchieved', 
                            'RSA_S1R0_signalerChoice', 'RSA_S1R0_receiverChoice', 'RSA_S1R0_modelRewardsAchieved', 
                            'RSA_S1R1_signalerChoice', 'RSA_S1R1_receiverChoice', 'RSA_S1R1_modelRewardsAchieved']
        output = pd.Series([s1r0_signalerChoice, s1r0_receiverChoice, s1r0_rewardsAchieved,
                            s1r1_signalerChoice, s1r1_receiverChoice, s1r1_rewardsAchieved,
                            s1r0RSA_signalerChoice, s1r0RSA_receiverChoice, s1r0RSA_rewardsAchieved,
                            s1r1RSA_signalerChoice, s1r1RSA_receiverChoice, s1r1RSA_rewardsAchieved], index=newColumnNames)
        return(output)

    def setupIWInference(self, dfRow):
        setupTrial = SetupMisyakTrial(rationalityParameter = self.modelParameters['alpha'], 
            valueOfReward = self.modelParameters['valueOfReward'], 
            signalMeaningPrior = self.formatSignalPrior(self.modelParameters['signalMeaningPrior']),
            silencePossible = True, 
            signalCost = self.modelParameters['costOfSignal'], 
            costOfPunishment = self.modelParameters['costOfPunishment'],
            nBoxes = self.modelParameters['nBoxes'])

        nSignalsIW, nAxesIW, shadowIW, nRewardsIW = self.getCondition(dfRow)
        _, getReceiver, getSignaler, getReceiverOne = setupTrial(nSignalsIW, nAxesIW, shadowIW, nRewardsIW, recOne = True)
        return(getReceiver, getSignaler, getReceiverOne)

    def setupRSAInference(self, dfRow):
        nSignalsIW, nAxesIW, shadowIW, nRewardsIW = self.getCondition(dfRow)
        setupRSASpeaker = SetupRSAMisyakTrial(rationalityParameter=self.modelParameters['alpha'], 
                        valueOfReward= self.modelParameters['valueOfReward'], 
                        signalMeaningPrior = self.formatSignalPrior(self.modelParameters['signalMeaningPrior']), 
                        silencePossible = True, 
                        signalCost = self.modelParameters['costOfSignal'], 
                        costOfPunishment = self.modelParameters['costOfPunishment'], 
                        nBoxes = self.modelParameters['nBoxes'])

        _, _, getRSASpeaker1 = setupRSASpeaker(nSignalsIW, nAxesIW, shadowIW, nRewardsIW)

        
        setupRSAReceiver = SetupRSAMisyakTrial_Receiver(rationalityParameter=self.modelParameters['alpha'], 
                        valueOfReward=self.modelParameters['valueOfReward'], 
                        signalMeaningPrior = self.formatSignalPrior(self.modelParameters['signalMeaningPrior']), 
                        silencePossible=True, 
                        signalCost = self.modelParameters['costOfSignal'], 
                        costOfPunishment = self.modelParameters['costOfPunishment'], 
                        nBoxes = 3)
        getRSAReceiver0, _, getRSAReceiver1 = setupRSAReceiver(nSignalsIW, nAxesIW, shadowIW, nRewardsIW)
        return(getRSAReceiver0, getRSASpeaker1, getRSAReceiver1)

    def getCondition(self, dfRow, maxAxes=2, maxTokens=2):
        nAxes = dfRow['n_axes']
        nTokens = dfRow['n_tokens']
        showShadow = dfRow['show_imprints']
        nRewards = dfRow['n_bananas']
        return(nTokens, nAxes, showShadow, nRewards)

    def sampleSignalerChoice(self, getSignaler, world, iw = True):
        if iw:
            signalPDF = getSignaler({'worlds':world})
            signal = signalPDF.sample(weights = signalPDF.columns[0]).index[0][0]
        else:
            signalPDF = getSignaler(world)
            signal = signalPDF.sample(weights = signalPDF.columns[0]).index[0]
        return(signal)

    def sampleReceiverChoice(self, getReceiver, signal, iw=True):
        receiverPDF = getReceiver(signal)

        if round(sum(receiverPDF[receiverPDF.columns[0]]),8) != 1.0:
            warnings.warn("signaler signal not consistent with anything, receiver samples randomly.")
            receiverChoice = receiverPDF.sample().index[0]
        else:
            receiverChoice = receiverPDF.sample(weights = receiverPDF.columns[0]).index[0]

        if iw:
            actionIndex = receiverPDF.index.names.index('actions')
            receiverAction = receiverChoice[actionIndex] 
        else:
            receiverAction = receiverChoice
        return(receiverAction)

    def sampleRSAReceiverActionFromInference(self, receiverInference, nAxes):
        actionSpace = [a for a in itertools.product([1,0], repeat = self.modelParameters['nBoxes']) if sum(a) <= nAxes]
        if receiverInference in actionSpace:
            selectedAction = receiverInference
        else:
            warnings.warn('inferred world has more rewards than receiver can achieve')
            # max rewards given the inferred world -- then sample uniformly from them
            predictedReward = [self.getRewardsAchieved(receiverInference, action) for action in actionSpace]
            bestPredictedActionIndices = np.argwhere(predictedReward == np.max(predictedReward)).flatten().tolist()
            selectedAction = actionSpace[np.random.choice(bestPredictedActionIndices)]
        return(selectedAction)

    def getRewardsAchieved(self, world, receiverAction):
        bananasAchieved = [box+ax for box, ax in zip(world, receiverAction)]
        return(bananasAchieved.count(2))

    def formatSignalPrior(self, probability, categoryNames = ['1', '-1'], decimalPlaces = 7):
        prior = {categoryNames[0]:round(probability, decimalPlaces), categoryNames[1]: round(1-probability, decimalPlaces)}
        return(prior)


class EvaluateMisyakDFSimulation():
    def __init__(self, fixedModelParameters, simulatedData):
        self.fixedModelParameters = fixedModelParameters
        self.simulatedData = simulatedData

    def __call__(self, alpha, meaningPrior):
        modelParameters = {'alpha':alpha, 
                    'valueOfReward': self.fixedModelParameters['valueOfReward'], 
                    'signalMeaningPrior': meaningPrior, 
                    'costOfSignal': self.fixedModelParameters['costOfSignal'], 
                    'costOfPunishment': self.fixedModelParameters['costOfPunishment'], 
                    'nBoxes': self.fixedModelParameters['nBoxes']}

        getMisyakModelBehavior = SimulateMisyakModelFromDF(modelParameters)
        modelSimulations = self.simulatedData.apply(getMisyakModelBehavior, axis=1)
        rewardsAchievedByStrategy = modelSimulations[['IW_S1R0_modelRewardsAchieved', 'IW_S1R1_modelRewardsAchieved', 'RSA_S1R0_modelRewardsAchieved', 'RSA_S1R1_modelRewardsAchieved']].mean()
        return(rewardsAchievedByStrategy)