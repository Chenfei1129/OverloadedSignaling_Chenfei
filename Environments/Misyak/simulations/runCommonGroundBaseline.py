import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import pandas as pd
import numpy as np

from Environments.Misyak.simulations.commonGroundManipulation import SimulateMisyakModelFromDF


def main():
	filename = './delme_test.pkl' #'./commonGroundBaselineManipulation_seed408.pkl'

	alpha = 3.0
	rewardValue = 5
	prior = .55
	sigCost = .01
	scorpionCost = -.5 

	simulatedTrials = pd.read_pickle('./commonGroundExptTrials.pkl')
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