import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Simulations.setupSampledTrials as targetCode


###########################################################################################################
##### Signal is Consistent - Experiment
###########################################################################################################	
@ddt
class TestSampleExperimentEnvironment(unittest.TestCase):
	def setUp(self):
		pass


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
