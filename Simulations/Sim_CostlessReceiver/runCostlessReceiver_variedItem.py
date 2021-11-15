import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd
import Simulations.modelLabels as ML
from Simulations.setupModelInferences_CostlessReceiver import SetupModels_CostlessReceiver
from Simulations.setupModelInferences import SetupModels
from Simulations.setupSampledTrials import SampleExperimentEnvironment
from Simulations.setupMetrics import SimulateModelFromDF, SingleAgentUtility_TaxicabMetric, CostlessReceiverUtility

def main():
	randomSeed = 423
	filename = './simulation_costlessReceiver_a4_r8_seed423_2-9ItemFullVocabActualNoPa.pkl'
	modNames = ['RSA_S0R0', 'RSA_S1R1']#'RSA_S0R0', 'RSA_S1R1', 'IW_S1R1'
	
	#Fixed Parameters
	alpha = 4
	rewardValue = 8
	params_a4r8 = {'rationality':alpha, 'reward': rewardValue}
	nruns = 500

	#Sampling spaces
	gridDims = (9,10) #0-8, by 0-9
	s = (4,0)
	r = (4,7)
	fixedEnvironmentParameters = {'gridSize': gridDims, 'signalerPosition': s, 'receiverPosition': r}
	
	itemSpace = ML.itemSpace_3Color3Shape
	signalSpace = ML.featureSpace_3Color3Shape
	fluidParameterSpaces = {'targets': itemSpace,'signals': signalSpace}

	#Parameters specific to this simualation:
	signalProp = 1.0
	prop = True

	# Trial Setup
	np.random.seed(randomSeed)
	getEnvironment = SampleExperimentEnvironment(fixedEnvironmentParameters, fluidParameterSpaces)
	simulatedTrials = pd.DataFrame(columns = ML.SAMPLED_TRIAL_PARAMETERS)

	for indx in range(nruns):
		nItems = np.random.choice(range(2,10))
		goal, sigSpace, targetD, nTargets, relevantSignalProp = getEnvironment(numberOfPossibleTargets = nItems, numberOfSignals = signalProp, vocabProportion = prop)
		simulatedTrials.loc[indx] = [s, r, goal, sigSpace, targetD, nTargets, relevantSignalProp]

	#Cost Receiver
	#setupModInference = SetupModels(modelParameters=params_a4r8, modelNames = modNames)
	getSAUtil = SingleAgentUtility_TaxicabMetric(reward = rewardValue)

	#getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility= getSAUtil)
	#rsaReceiverSimulation = simulatedTrials.apply(getModelMetrics, axis=1)
	#costReceiverWithTrialInfo = simulatedTrials.join(rsaReceiverSimulation)

	#Costless Receiver
	setupModInference_C0R = SetupModels_CostlessReceiver(modelParameters=params_a4r8, modelNames = modNames)
	getReceiverUtil = CostlessReceiverUtility(reward = rewardValue)

	getModelMetrics_R0 = SimulateModelFromDF(buildModels=setupModInference_C0R, getUtility= getSAUtil, receiverSpecificUtility = getReceiverUtil)
	costlessReceiverSimulation = simulatedTrials.apply(getModelMetrics_R0, axis=1)
	

	fullSimulations = simulatedTrials.join(costlessReceiverSimulation, rsuffix='_C0R')
	#fullSimulations = costReceiverWithTrialInfo.join(costlessReceiverSimulation, rsuffix='_C0R')

	#Save full simulated data
	fullSimulations.to_pickle(filename)

if __name__ == "__main__":
	main()
