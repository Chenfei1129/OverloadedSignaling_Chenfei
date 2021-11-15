import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import re

import Algorithms.constantNames as NC

from Environments.Experiment.experimentConstruction import JointActionUtility_CostRatioReceiver

from Environments.Experiment.setupInference_Experiment import SetupExperiment_SignalsSeparated
from Algorithms.RSA.SetupRSAInference import SetupExperiment_RSAListenerInference
from Algorithms.RSA.RSAExtensionForCostRatio import SpeakerActionSignalDistribution_NoReceiverCost, SetupExperiment_RSASpeakerWithActionChoice_NoReceiverCost
from Algorithms.JointUtility.utilityOnlyModel import UtilityDrivenSignaler_NoReceiverCosts, UtilityDrivenReceiver, UtilityDrivenReceiver_CostRatio, UtilityDrivenSignaler_RatioReceiverCosts

"""
modelParameters: rationality and reward (dictionary {'rationality': real number, 'reward': real number})
modelNames: list of models to create for the simulation of the form MODELTYPE_SPEAKERLEVELRECEIVERLEVEL_OTHERADDONS. 
    model types: IW, RSA, JU
    speaker levels: For RSA and IW can be S0, S1, S2, S3 (and same for R)

"""

class SetupModels_CostRatioReceiver():
    def __init__(self, modelParameters, modelNames):
        self.modelParameters = modelParameters
        self.modelNames = modelNames
        
        self.signalerInactionPossible = False
        if 'signalerInaction' in modelParameters.keys():
            self.signalerInactionPossible = modelParameters['signalerInaction']
          
    def __call__(self, environmentParameters):
        modelInferenceDictionary = {}
        rsaModels = [name for name in self.modelNames if 'RSA' in name]
        if len(rsaModels) > 0:
            modelLevels = {modelName : [x[-1:] for x in re.findall(r"[.]*_S[0-9]|R[0-9]", modelName)] + [('LO' in modelName)] 
                                        for modelName in rsaModels}
            rsaRInferences, rsaSInferences, rsaSInferences_LO = self.setupRSAInference(environmentParameters, modelLevels)
            formatModelInference = lambda levels: [rsaSInferences_LO[levels[0]] if levels[2] else rsaSInferences[levels[0]], rsaRInferences[levels[1]]]

            rsaModelDict = {model: formatModelInference(levels) for model, levels in modelLevels.items()}
            modelInferenceDictionary.update(rsaModelDict)

        iwModels = [name for name in self.modelNames if 'IW' in name]
        if len(iwModels) > 0:
            #iwSpeakerLevels = [speaker for mod in iwModels for speaker in speakers if speaker in mod]
            #iwReceiverLevels = [receiver for mod in iwModels for receiver in receivers if receiver in mod]
            receiverN_IW, signalerP_IW_actions, receiverP_IW = self.setupIWInference(environmentParameters)
            modelInferenceDictionary['IW_S1R0'] = [signalerP_IW_actions, receiverN_IW]
            modelInferenceDictionary['IW_S1R1'] = [signalerP_IW_actions, receiverP_IW]

        juModels = [name for name in self.modelNames if 'JU' in name]
        if len(juModels) > 0:
            if len(juModels) != 1:
                warnings.warn(">1 Joint Utility model was specified but only one was created")

            receiver_JU, signaler_JU = self.setupJointUtilityInference(environmentParameters)
            juModelName = juModels[0]
            modelInferenceDictionary[juModelName] = [signaler_JU, receiver_JU]  

        return(modelInferenceDictionary)
        
    def setupIWInference(self, sampledEnvironmentParameters):
        runIWInference = SetupExperiment_SignalsSeparated(self.modelParameters['rationality'], self.modelParameters['reward'],   JointActionUtility_CostRatioReceiver, receiverCostRatio = self.modelParameters['receiverCostRatio'],
                                                         signalerInactionPossible=self.signalerInactionPossible , signalCost = self.calculateSignalCost ) ########
        receiverNaive_IW, signaler_IWAction, receiverPragmatic_IW = runIWInference(sampledEnvironmentParameters['signalerLocation'], 
                                                           sampledEnvironmentParameters['receiverLocation'], 
                                                           sampledEnvironmentParameters['signals'], 
                                                           sampledEnvironmentParameters['targetDictionary']
                                                                                  )
        return(receiverNaive_IW, signaler_IWAction, receiverPragmatic_IW)


    def setupRSAInference(self, sampledEnvironmentParameters, modelLevels):
        #print(modelLevels.values())
        maxReceiver = max([int(rec) for sig, rec, lo in modelLevels.values()])#
        rsaReceiverInferences = self.setupRSAListeners(sampledEnvironmentParameters, maxReceiver)
        
        if any([not lo for sig, rec, lo in modelLevels.values()]):
            maxSignaler = max([int(sig) for sig, rec, lo in modelLevels.values() if not lo])
        else:
            maxSignaler = None
        if any([lo for sig, rec, lo in modelLevels.values()]):
            maxLanguageOnlyModel = max([int(sig) for sig, rec, lo in modelLevels.values() if lo])
        else:
            maxLanguageOnlyModel = None
        rsaActingSpeakers, rsaLanguageSpeakers = self.setupRSASpeakers(sampledEnvironmentParameters, maxSignaler, maxLanguageOnlyModel)
        return(rsaReceiverInferences, rsaActingSpeakers, rsaLanguageSpeakers)

    def setupRSAListeners(self, sampledEnvironmentParameters, maxReceiver):
        #Setup Uniform Target Prior
        targetSpace = list(sampledEnvironmentParameters['targetDictionary'].values())
        uniformTargetPrior = {target: 1/len(targetSpace) for target in targetSpace}

        runRSAListenerInference = SetupExperiment_RSAListenerInference(self.modelParameters['rationality'])
        rsaListeners = runRSAListenerInference(targetPrior=uniformTargetPrior, 
                                                signalSpace=sampledEnvironmentParameters['signals'],
                                                maxLayer = maxReceiver,
                                                getAllLayers=True)
        return(rsaListeners)
    
    def setupRSASpeakers(self, sampledEnvironmentParameters, maxActingSpeakerLevel, maxLanguageOnlySpeakerLevel):
        targetSpace = list(sampledEnvironmentParameters['targetDictionary'].values())
        uniformTargetPrior = {target: 1/len(targetSpace) for target in targetSpace} #Setup Uniform Target Prior
        rsaActingSpeakers = None
        rsaLanguageSpeakers = None

        if maxActingSpeakerLevel is not None:
            runRSASpeakerInference = SetupExperiment_RSASpeakerWithActionChoice_NoReceiverCost(self.modelParameters['rationality'], self.modelParameters['reward'],receiverCostRatio = self.modelParameters['receiverCostRatio'],
                                                                                              signalerInactionPossible= self.signalerInactionPossible ) #########3
            rsaActingSpeakers = runRSASpeakerInference(targetPrior=uniformTargetPrior, 
                                               signalSpace=sampledEnvironmentParameters['signals'], 
                                               targetDictionary=sampledEnvironmentParameters['targetDictionary'], 
                                               signalerLocation=sampledEnvironmentParameters['signalerLocation'], 
                                               receiverLocation=sampledEnvironmentParameters['receiverLocation'], 
                                               maxLayer = maxActingSpeakerLevel,
                                               getAllLayers = True)
        if maxLanguageOnlySpeakerLevel is not None:
            runRSASpeakerInference = SetupExperiment_RSASpeakerInference(self.modelParameters['rationality'])
            rsaLanguageSpeakers = runRSASpeakerInference(targetPrior=uniformTargetPrior, 
                                               signalSpace=sampledEnvironmentParameters['signals'],
                                               maxLayer = maxLanguageOnlySpeakerLevel,
                                               getAllLayers = True)
        return(rsaActingSpeakers, rsaLanguageSpeakers)
   
    def setupJointUtilityInference(self, sampledEnvironmentParameters):
        receiver_JointUtility = UtilityDrivenReceiver_CostRatio(signalerLocation=sampledEnvironmentParameters['signalerLocation'], 
                                          receiverLocation=sampledEnvironmentParameters['receiverLocation'], 
                                          targetDictionary=sampledEnvironmentParameters['targetDictionary'], 
                                          valueOfReward=self.modelParameters['reward'], 
                                          rationality=self.modelParameters['rationality'], receiverCostRatio = self.modelParameters['receiverCostRatio'] )
        
        signaler_JointUtility = UtilityDrivenSignaler_RatioReceiverCosts(signalSpace=sampledEnvironmentParameters['signals'], ##############
                                          signalerLocation=sampledEnvironmentParameters['signalerLocation'], 
                                          receiverLocation=sampledEnvironmentParameters['receiverLocation'], 
                                          targetDictionary=sampledEnvironmentParameters['targetDictionary'], 
                                          valueOfReward=self.modelParameters['reward'], 
                                          rationality=self.modelParameters['rationality'], receiverRatioCost = self.modelParameters['receiverCostRatio'], signalerInactionPossible= self.signalerInactionPossible ) 
        return(receiver_JointUtility, signaler_JointUtility)

