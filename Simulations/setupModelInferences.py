import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))

import warnings
import re
from Environments.Experiment.costSignal import  CalculateSignalCost#calculateSignalCost,
from Environments.Experiment.setupInference_Experiment import SetupExperiment_SignalsSeparated_Levels
from Algorithms.RSA.SetupRSAInference import SetupExperiment_RSASpeakerWithActionChoice, SetupExperiment_RSAListenerInference, SetupExperiment_RSASpeakerInference
from Algorithms.JointUtility.utilityOnlyModel import UtilityDrivenSignaler, UtilityDrivenReceiver

"""
modelParameters: rationality and reward (dictionary {'rationality': real number, 'reward': real number})
modelNames: list of models to create for the simulation of the form MODELTYPE_SPEAKERLEVELRECEIVERLEVEL_OTHERADDONS. 
    model types: IW, RSA, JU
    speaker levels: For RSA and IW can be S0, S1, S2, S3 (and same for R)

"""

class SetupModels():
    def __init__(self, modelParameters, modelNames):
        self.modelNames = modelNames
        self.rewardValue = modelParameters['reward']
        self.rationality = modelParameters['rationality']
        self.costFunction = modelParameters['costFunction']
        self.signalCost = modelParameters['signalCost']
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
            modelLevels = {modelName : [x[-1:] for x in re.findall(r"[.]*_S[0-9]|R[0-9]", modelName)] for modelName in iwModels}
            iwSInferences, iwRInferences = self.setupIWInference(environmentParameters, modelLevels)
            print("IW Inferences have finished being constructed \n")
            formatModelInferenceIW = lambda levels: [iwSInferences[levels[0]], iwRInferences[levels[1]]]
            iwModelDict = {modelName: formatModelInferenceIW(levels) for modelName, levels in modelLevels.items()}
            modelInferenceDictionary.update(iwModelDict)

        juModels = [name for name in self.modelNames if 'JU' in name]
        if len(juModels) > 0:
            if len(juModels) != 1:
                warnings.warn(">1 Joint Utility model was specified but only one was created")

            receiver_JU, signaler_JU = self.setupJointUtilityInference(environmentParameters)
            juModelName = juModels[0]
            modelInferenceDictionary[juModelName] = [signaler_JU, receiver_JU]  
        return(modelInferenceDictionary)
        
    def setupIWInference(self, sampledEnvironmentParameters, modelLayers):
        maxModLayer = max([max(int(rec), int(sig)) for sig, rec in modelLayers.values()])
        runIWInference = SetupExperiment_SignalsSeparated_Levels(###
                                            beta = self.rationality, 
                                            valueOfReward = self.rewardValue, 
                                            getCost = self.costFunction, signalCost = CalculateSignalCost(self.signalCost) ,  ###
                                            signalerInactionPossible=self.signalerInactionPossible)

        signalerDictIW, receiverDictIW = runIWInference(sampledEnvironmentParameters['signalerLocation'], 
                                                           sampledEnvironmentParameters['receiverLocation'], 
                                                           sampledEnvironmentParameters['signals'], 
                                                           sampledEnvironmentParameters['targetDictionary'], 
                                                           maxLayer = maxModLayer)
        return(signalerDictIW, receiverDictIW)


    def setupRSAInference(self, sampledEnvironmentParameters, modelLevels):
        maxReceiver = max([int(rec) for sig, rec, lo in modelLevels.values()])
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

        runRSAListenerInference = SetupExperiment_RSAListenerInference(self.rationality)
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
            runRSASpeakerInference = SetupExperiment_RSASpeakerWithActionChoice(
                                                rationality = self.rationality, 
                                                valueOfReward = self.rewardValue,
                                                getActionCost = self.costFunction,###
                                                signalerInactionPossible= self.signalerInactionPossible)

            rsaActingSpeakers = runRSASpeakerInference(targetPrior=uniformTargetPrior, 
                                               signalSpace=sampledEnvironmentParameters['signals'], 
                                               targetDictionary=sampledEnvironmentParameters['targetDictionary'], 
                                               signalerLocation=sampledEnvironmentParameters['signalerLocation'], 
                                               receiverLocation=sampledEnvironmentParameters['receiverLocation'], 
                                               maxLayer = maxActingSpeakerLevel,
                                               getAllLayers = True)
        if maxLanguageOnlySpeakerLevel is not None:
            runRSASpeakerInference = SetupExperiment_RSASpeakerInference(self.rationality)
            rsaLanguageSpeakers = runRSASpeakerInference(targetPrior=uniformTargetPrior, 
                                               signalSpace=sampledEnvironmentParameters['signals'],
                                               maxLayer = maxLanguageOnlySpeakerLevel,
                                               getAllLayers = True)
        return(rsaActingSpeakers, rsaLanguageSpeakers)
   
    def setupJointUtilityInference(self, sampledEnvironmentParameters):
        receiver_JointUtility = UtilityDrivenReceiver(signalerLocation=sampledEnvironmentParameters['signalerLocation'], 
                                      receiverLocation=sampledEnvironmentParameters['receiverLocation'], 
                                      targetDictionary=sampledEnvironmentParameters['targetDictionary'], 
                                      valueOfReward=self.rewardValue, 
                                      rationality=self.rationality,
                                      costFunction = self.costFunction)
    
        signaler_JointUtility = UtilityDrivenSignaler(signalSpace=sampledEnvironmentParameters['signals'], 
                                      signalerLocation=sampledEnvironmentParameters['signalerLocation'], 
                                      receiverLocation=sampledEnvironmentParameters['receiverLocation'], 
                                      targetDictionary=sampledEnvironmentParameters['targetDictionary'], 
                                      valueOfReward=self.rewardValue, 
                                      rationality=self.rationality, 
                                      costFunction = self.costFunction,
                                      signalerInactionPossible = self.signalerInactionPossible) 

        return(receiver_JointUtility, signaler_JointUtility)

