import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import *
from Environments.Grosse.grosseConstruction import *
import Environments.Grosse.consistentSignalChecks_Grosse as targetCode


###########################################################################################################
##### Signal is Consistent - Grosse
###########################################################################################################	
@ddt
class TestSignalIsConsistentGrosse(unittest.TestCase):
	def setUp(self):
		pass

	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'R')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('R', 'L')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_ConsistentActionAmbiguousSignal(self, mind, signalerType):
		signal = 'help'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)


	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'n')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('L', 'n')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_InconsistentActionAmbiguousSignal(self, mind, signalerType):
		signal = 'help'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)


	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'n', NC.ACTIONS:('R', 'L')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_InconsistentIntention(self, mind, signalerType):
		signal = 'help'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)


	@data(('', {NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'n')}, '1'),
		('null', {NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('L', 'n')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_ConsistentActionNullSignal(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)


	@data(('', {NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L')}, '1'),
		('me', {NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('L', 'R')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_InconsistentActionNullSignal(self, signal, mind, signalerType):
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)


	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L')}, '1', True), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'R')}, '1', False), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('R', 'L')}, '1', True))
	@unpack
	def test_signalIsConsistent_Grosse_LeftSignal(self, mind, signalerType, expectedResult):
		signal = 'help Left'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertEqual(isSignalConsistent, expectedResult)


	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'R')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('R', 'L')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_InconsistentRightSignal(self, mind, signalerType):
		signal = 'help Right'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)


	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'either', NC.ACTIONS:('n', 'L')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'either', NC.ACTIONS:('n', 'R')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'either', NC.ACTIONS:('R', 'L')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_ConsistentHelpEitherSignal(self, mind, signalerType):
		signal = 'help either'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertTrue(isSignalConsistent)


	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'R', NC.ACTIONS:('n', 'R')}, '1'), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'LR', NC.ACTIONS:('R', 'L')}, '1'))
	@unpack
	def test_signalIsConsistent_Grosse_InconsistentHelpEitherSignal(self, mind, signalerType):
		signal = 'help either'
		isSignalConsistent = targetCode.signalIsConsistent_Grosse(signal, mind, signalerType)
		self.assertFalse(isSignalConsistent)

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest.main(verbosity=2)
