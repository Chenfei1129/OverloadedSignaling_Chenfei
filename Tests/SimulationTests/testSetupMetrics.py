import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Simulations.setupMetrics as targetCode

from Simulations.setupModelInferences import SetupModels
#from Simulations.setupSampledTrials import SampleExperimentEnvironment
#from Simulations.setupMetrics import SimulateModelFromDF, SingleAgentUtility_TaxicabMetric
#from Environments.Experiment.experimentConstruction import calculateLocationCost_TaxicabMetric

###########################################################################################################
##### Signal is Consistent - Experiment
###########################################################################################################	
@ddt
class TestDataFrameSimulation(unittest.TestCase):
	def setUp(self):
		gridDims = (9,10) #0-8, by 0-9
		s = (4,0)
		r = (4,7)

		environmentParameters= {'observation': {'intentions':'red square'}, 
                        'signals': ['green', 'purple', 'triangle', 'square', 'circle', 'red'], 
                        'targetDictionary':{(0, 9): 'green circle',
                         (1, 1): 'purple circle',
                         (1, 6): 'red circle',
                         (1, 8): 'green square',
                         (5, 6): 'purple square',
                         (7, 7): 'green triangle',
                         (8, 9): 'red square'}, 
                        'signalerLocation':s, 'receiverLocation':r} #dictionary to pandas df.

		params_a4r15 = {'rationality':4, 'reward': 8, 'costFunction': calculateLocationCost_TaxicabMetric, 'signalerInaction': True}
		modNames = ['RSA_S0R0', 'RSA_S1R1', 'IW_S1R0', 'IW_S1R1', 'JU']
		setupModInference = SetupModels(modelParameters=params_a4r15, modelNames = modNames)
		getSAUtil = SingleAgentUtility_TaxicabMetric(reward = params_a4r15['reward'])
		getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility= getSAUtil)
		regularReceiverSimulation = simulatedTrials.apply(getModelMetrics, axis=1)


	@data()
	@unpack
	def test_dfSetup(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertTrue(consistencyBoolean)

	def tearDown(self):
		pass



if __name__ == '__main__':
	unittest.main(verbosity=2)
