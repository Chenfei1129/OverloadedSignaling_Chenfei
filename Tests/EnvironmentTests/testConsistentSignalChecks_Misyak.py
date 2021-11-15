import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import *
from Environments.Misyak.misyakConstruction import *
import Environments.Misyak.consistentSignalChecks_Misyak as targetCode

@ddt
class TestSignalConsistent(unittest.TestCase):
	def setUp(self): 
		pass

	###########################################################################################################
	##### Signal is Consistent - Misyak
	###########################################################################################################
	@data(((1,0,0), {NC.WORLDS:(1,0,0)}, "1"),
		((1,1,0), {NC.WORLDS:(1,1,0)}, "1"))
	@unpack
	def test_signalIsConsistent_Boxes_ConsistentExactSignal_GoToTypeSignaler(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Boxes(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)

	@data(((1,1,0), {NC.WORLDS:(0,0,1)}, "-1"), 
		((1,0,0), {NC.WORLDS:(0,1,1)}, "-1"))
	@unpack
	def test_signalIsConsistent_Boxes_ConsistentFullExactSignal_AviodTypeSignaler(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Boxes(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)

	@data(((1,0,0), {NC.WORLDS:(0,1,0)}, "-1"), 
		((1,0,0), {NC.WORLDS:(1,1,0)}, "1"))
	@unpack
	def test_signalIsConsistent_Boxes_ConsistentPartialSignal(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Boxes(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)

	@data(((0,0,0), {NC.WORLDS:(1,0,0)}, "1"), 
		((0,0,0), {NC.WORLDS:(1,0,0)}, "-1"))
	@unpack
	def test_signalIsConsistent_Boxes_NullSignals(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Boxes(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)

	@data(((1,0,0), {NC.WORLDS:(1,0,0)}, "-1"), 
		((1,0,0), {NC.WORLDS:(0,1,0)}, "1"))
	@unpack
	def test_signalIsConsistent_Boxes_InconsistentSignals(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Boxes(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)

	@data(((1,1,0), {NC.WORLDS:(1,0,0)}, "-1"), 
	((1,1,0), {NC.WORLDS:(1,0,0)}, "1"))
	@unpack
	def test_signalIsConsistent_Boxes_InconsistentMixedTokenMeaning(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Boxes(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)


	def tearDown(self):
			pass


if __name__ == '__main__':
	unittest.main(verbosity=2)