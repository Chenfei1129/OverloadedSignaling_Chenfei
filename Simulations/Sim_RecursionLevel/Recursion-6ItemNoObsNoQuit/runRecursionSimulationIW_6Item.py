import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd
import itertools
import warnings

import Algorithms.constantNames as NC
import Simulations.modelLabels as ML

from Simulations.setupModelInferences import SetupModels
from Simulations.setupSampledTrials import SampleExperimentEnvironment
from Simulations.setupMetrics import SimulateModelFromDF, SingleAgentUtility_TaxicabMetric


def main():
	randomSeed = 195
	filename = './simulation_recursion_a4_r8_seed195_6ItemFullVocabIW_100Runs_MaxLayer2.pkl'
	
	#Fixed Parameters
	alpha = 4	
	rewardValue = 8
	nruns = 100

	#Sampling spaces
	gridDims = (9,10) #0-8, by 0-9
	s = (4,0)
	r = (4,7)
	fixedEnvironmentParameters = {'gridSize': gridDims, 'signalerPosition': s, 'receiverPosition': r}
	
	itemSpace = ML.itemSpace_3Color3Shape
	signalSpace = ML.featureSpace_3Color3Shape
	fluidParameterSpaces = {'targets': itemSpace,'signals': signalSpace}

	#Parameters specific to this simualation:
	nItems = 6
	signalProp = 1.0
	prop = True

	# Trial Setup
	np.random.seed(randomSeed)
	getEnvironment = SampleExperimentEnvironment(fixedEnvironmentParameters, fluidParameterSpaces)
	simulatedTrials = pd.DataFrame(columns = ML.SAMPLED_TRIAL_PARAMETERS)

	for indx in range(nruns):
		#nItems = np.random.choice(range(2,10))
		goal, sigSpace, targetD, nTargets, relevantSignalProp = getEnvironment(numberOfPossibleTargets = nItems, numberOfSignals = signalProp, vocabProportion = prop)
		simulatedTrials.loc[indx] = [s, r, goal, sigSpace, targetD, nTargets, relevantSignalProp]

	params_a4r8 = {'rationality':alpha, 'reward': rewardValue}
	modNames = ['IW_S1R0', 'IW_S1R1', 'IW_S1R2', 
				'IW_S2R0', 'IW_S2R1', 'IW_S2R2']
	setupModInference = SetupModels(modelParameters=params_a4r8, modelNames = modNames)
	getSAUtil = SingleAgentUtility_TaxicabMetric(reward = rewardValue)

	getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility= getSAUtil)
	rsaReceiverSimulation = simulatedTrials.apply(getModelMetrics, axis=1)
	fullSimulations = simulatedTrials.join(rsaReceiverSimulation)

	#Save full simulated data
	fullSimulations.to_pickle(filename)

if __name__ == "__main__":
	main()