import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pandas as pd

from Simulations.setupModelInferences import SetupModels
from Simulations.setupSampledTrials import SampleExperimentEnvironment
from Simulations.setupMetrics import SimulateModelFromDF, SingleAgentUtility_TaxicabMetric


def main():
	alpha = 4	
	rewardValue = 8
	
	filename = './simulation_checkAgainstFVByItemSim.pkl'
	simulatedTrials = pd.read_pickle('/home/stacyste/Documents/Research/OverloadedSignaling/Simulations/Sim_ItemByFullVPartialVocab_Combined/simulation_a4_r8_fullVocab_2000Runs_seed101.pkl')
	
	params_a4r8 = {'rationality':alpha, 'reward': rewardValue}
	modNames = ['IW_S1R0', 'IW_S1R1', 'RSA_S1R1']
	setupModInference = SetupModels(modelParameters=params_a4r8, modelNames = modNames)
	getSAUtil = SingleAgentUtility_TaxicabMetric(reward = rewardValue)

	getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility= getSAUtil)
	rsaReceiverSimulation = simulatedTrials.apply(getModelMetrics, axis=1)
	fullSimulations = simulatedTrials.join(rsaReceiverSimulation, rsuffix = "_R")

	#Save full simulated data
	fullSimulations.to_pickle(filename)

if __name__ == "__main__":
	main()