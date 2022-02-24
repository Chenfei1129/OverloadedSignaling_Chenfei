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
from Simulations.setupMetrics import SimulateModelFromDF, SingleAgentUtility_CustomCostFunction

from Environments.Experiment.barrierConstruction.utilityFromPolicyDictionary import SetupPolicyTableForEnvironment, SetupIndividualActionCost
from Environments.Experiment.barrierConstruction.transitionTable import createTransitionTable


def main():
	randomSeed = 2802
	filename = './simulation_recursion_obstacleHNearSig_SQuit_a4_r8_seed2802_6ItemFullVocab_300Runs.pkl'
	
	#Parameters
	signalerCanQuit = True
	alpha = 4
	rewardValue = 8
	nruns = 300
	nItems = 6

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
		#nItems = np.random.choice(range(2,10))
		goal, sigSpace, targetD, nTargets, relevantSignalProp = getEnvironment(numberOfPossibleTargets = nItems, numberOfSignals = signalProp, vocabProportion = prop)
		simulatedTrials.loc[indx] = [s, r, goal, sigSpace, targetD, nTargets, relevantSignalProp]

	#Obstacle Policy calculations
	stateSet = list(itertools.product(range(gridDims[0]), range(gridDims[1])))
	actionSet = [(1,0), (-1,0), (0,1), (0,-1)]
	getPolicyDict = SetupPolicyTableForEnvironment(stateSet, actionSet)
	barrierHNearSig = [((3, 2), (0, 1)),
					 ((4, 2), (0, 1)),
					 ((5, 2), (0, 1)),
					 ((6, 2), (0, 1)),
					 ((7, 2), (0, 1)),
					 ((8, 2), (0, 1)),
					 ((3, 3), (0, -1)),
					 ((4, 3), (0, -1)),
					 ((5, 3), (0, -1)),
					 ((6, 3), (0, -1)),
					 ((7, 3), (0, -1)),
					 ((8, 3), (0, -1))]
	policiesBarrierHNS = getPolicyDict(barrierHNearSig)
	getTransitionTable = createTransitionTable(gridDims[0], gridDims[1], actionSet)
	transitionTableBarrierHNS = getTransitionTable(barrierHNearSig)

	getCostObstacleHNS = SetupIndividualActionCost(policiesBarrierHNS, transitionTableBarrierHNS, 1)

	# Modeling inference setup
	params_a4r8 = {'rationality' : alpha, 'reward': rewardValue, 'costFunction' : getCostObstacleHNS,'signalerInaction': signalerCanQuit}
	modNames = ['IW_S1R0', 'IW_S1R1', 'IW_S1R2', 
				'IW_S2R0', 'IW_S2R1', 'IW_S2R2',
				'RSA_S1R0', 'RSA_S1R1', 'RSA_S1R2', 
				'RSA_S2R0', 'RSA_S2R1', 'RSA_S2R2']
	setupModInference = SetupModels(modelParameters=params_a4r8, modelNames = modNames)
	getSAUtil_BarrierHNS = SingleAgentUtility_CustomCostFunction(reward = rewardValue, getCostFunction = getCostObstacleHNS)

	#Simulation step
	getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility=getSAUtil_BarrierHNS, signalerCanQuit=signalerCanQuit)
	rsaReceiverSimulation = simulatedTrials.apply(getModelMetrics, axis=1)
	fullSimulations = simulatedTrials.join(rsaReceiverSimulation)

	#Save full simulated data
	fullSimulations.to_pickle(filename)

if __name__ == "__main__":
	main()

