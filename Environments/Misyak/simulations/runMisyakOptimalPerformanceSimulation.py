import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import pandas as pd
import numpy as np

from Environments.Misyak.simulations.simulateModel_Misyak import SimulateMisyakModelFromDF


def main():
	filename = './modelPerformanceSimulation-2020-9-16.pkl'

	alpha = 10.0
	rewardValue = 1
	prior = .55
	sigCost = 0
	scorpionCost = -1

	simulatedTrials = pd.read_pickle('../data/conditionsForCommonGround2020-8-25.pkl')
	modelParams = {'alpha':alpha, 
					'valueOfReward': rewardValue, 
					'signalMeaningPrior': prior, 
					'costOfSignal': sigCost, 
					'costOfPunishment': scorpionCost, 
					'nBoxes': 3}

	np.random.seed(408)
	getMisyakModelBehavior = SimulateMisyakModelFromDF(modelParams)
	modelSimulations = simulatedTrials.apply(getMisyakModelBehavior, axis=1)
	fullSimulations = simulatedTrials.join(modelSimulations)

	#Save full simulated data
	fullSimulations.to_pickle(filename)

if __name__ == "__main__":
	main()