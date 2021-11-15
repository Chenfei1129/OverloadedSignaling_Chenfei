import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC

from Environments.Misyak.misyakConstruction import *
from Environments.Misyak.consistentSignalChecks_Misyak import signalIsConsistent_Boxes
from Environments.Grosse.grosseConstruction import *
from Environments.Grosse.consistentSignalChecks_Grosse import signalIsConsistent_Grosse
import Algorithms.ImaginedWe.GenerativeSignaler as targetCode


@ddt
class TestGenerativeSignaler(unittest.TestCase):
	def setUp(self): 
		#signal spaces
		oneTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 1)
		twoTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 2)

		# signal Cost schemes:
		noCosts = 0
		getSignalCost_noCosts = SignalCost_Misyak(noCosts)

		unevenCosts = (0, 1, 1.5)
		getSignalCost_unevenCosts = SignalCost_Misyak(unevenCosts)
		
		#signaler zero - Miysak
		self.getSignaler0_TwoToken = targetCode.SignalerZero(twoTokenSignalSpace, signalIsConsistent_Boxes, getSignalCost_noCosts)
		self.getSignaler0_OneToken = targetCode.SignalerZero(oneTokenSignalSpace, signalIsConsistent_Boxes, getSignalCost_noCosts)

		self.getSignaler0_TwoToken_unevenSignalCosts = targetCode.SignalerZero(twoTokenSignalSpace, signalIsConsistent_Boxes, getSignalCost_unevenCosts)
		self.getSignaler0_OneToken_unevenSignalCosts = targetCode.SignalerZero(oneTokenSignalSpace, signalIsConsistent_Boxes, getSignalCost_unevenCosts)

		# signaler zero - Grosse
		self.getSignaler0_battery = targetCode.SignalerZero(['you', ''], signalIsConsistent_Grosse)

	#tests the Misyak setup -- only the inputs for signals and worlds will determine the likelihood
	@data(({NC.WORLDS:(1,0,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS:(0,0,0), NC.SIGNALS:(1,0,0)}, '1', [.5, .5]), 
		({NC.WORLDS:(1,1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS:(0,0,0), NC.SIGNALS:(1,0,0)}, '1', [1.0/4, 1.0/3]), 
		({NC.WORLDS:(1,0,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS:(0,0,0), NC.SIGNALS:(0,1,0)}, '1', [0, 0]), 
		({NC.WORLDS:(1,0,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS:(0,0,0), NC.SIGNALS:(1,0,0)}, '-1', [0, 0]), 
		({NC.WORLDS:(1,1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS:(0,0,0), NC.SIGNALS:(0,0,1)}, '-1', [.5, .5]))
	@unpack
	def test_getSignalLikelihoodGivenMind(self, mind, signalerType, expectedResult):
		#generate an example multiindex condition of the mind + signal
		mindLabels = list(mind.keys())
		mindValues = [[v] for v in mind.values()]
		idx = pd.MultiIndex.from_product(mindValues, names=mindLabels)
		worldSignalCondition = pd.DataFrame(index=idx)

		likelihood = self.getSignaler0_TwoToken.getSignalLikelihoodGivenMind(worldSignalCondition, signalerType)
		likelihoodOneSignal = self.getSignaler0_OneToken.getSignalLikelihoodGivenMind(worldSignalCondition, signalerType)

		self.assertEqual(likelihood, expectedResult[0])
		self.assertEqual(likelihoodOneSignal, expectedResult[1])

	#tests the Grosse setup
	@data(({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L'), NC.SIGNALS: 'you'}, 1, 1.0), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('n', 'L'), NC.SIGNALS: ''}, 1, 0), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('R', 'L'), NC.SIGNALS: 'you'}, 1, 1.0), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('R', 'L'), NC.SIGNALS: ''}, 1, 0),
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('L', 'n'), NC.SIGNALS: 'you'}, 1, 0), 
		({NC.WORLDS: 'LR', NC.DESIRES: 1, NC.INTENTIONS: 'L', NC.ACTIONS:('L', 'n'), NC.SIGNALS: ''}, 1, 1.0))
	@unpack
	def test_getSignalLikelihoodGivenMind_Grosse(self, mind, signalerType, expectedResult):
		#generate an example multiindex condition of the mind + signal
		mindLabels = list(mind.keys())
		mindValues = [[v] for v in mind.values()]
		idx = pd.MultiIndex.from_product(mindValues, names=mindLabels)
		worldSignalCondition = pd.DataFrame(index=idx)

		likelihood = self.getSignaler0_battery.getSignalLikelihoodGivenMind(worldSignalCondition, signalerType)
		self.assertEqual(likelihood, expectedResult)

	# expected result 0/1: two token/one token rescaled probability, 2/3: two token/one token rescaled probability, uneven signal costs
	# examples 1 tests null signal - should be maximum of the allowable probability range
	# examples 2, 3 ,4  test one token signals
	# examples 5, 6, 7 test two token signals (where allowable)
	# example 8, 9 test one one token and one two token case for a different deviation allowance
	@data(((0,0,0), .05, [1,1, .15, .2625]), 
		((1,0,0), .05, [1,1,.15, .2625]), ((0,1,0), .05, [1,1,0.14428571428571427, 0.24583333333333332]), ((0,0,1), .05, [1,1, 0.14142857142857143,.2375]), 
		((1,1,0), .05, [1, None,0.14428571428571427, None]), ((1,0,1), .05, [1, None,0.14142857142857143, None]), ((0,1,1), .05, [1,None,0.1357142857142857,None]), 
		((1,0,1), .1, [1,None,0.13999999999999999,None]), ((0,1,0), .1, [1,1,0.14571428571428569,0.24166666666666667]))
	@unpack
	def test_rescaleSignalUtilityForCost(self, signal, factorOfDeviationFromUniform, expectedResult):
		mind = None

		rescaledSignalUtiltiyTwoToken = self.getSignaler0_TwoToken.rescaleSignalUtilityForCost(signal, mind, factorOfDeviationFromUniform)
		rescaledSignalUtiltiyTwoToken_unevenCosts = self.getSignaler0_TwoToken_unevenSignalCosts.rescaleSignalUtilityForCost(signal, mind, factorOfDeviationFromUniform)
		self.assertAlmostEqual(rescaledSignalUtiltiyTwoToken, expectedResult[0])
		self.assertAlmostEqual(rescaledSignalUtiltiyTwoToken_unevenCosts, expectedResult[2])

		if signal.count(1) <=1:
			rescaledSignalUtiltiyOneToken = self.getSignaler0_OneToken.rescaleSignalUtilityForCost(signal, mind, factorOfDeviationFromUniform)
			rescaledSignalUtiltiyOneToken_unevenCosts = self.getSignaler0_OneToken_unevenSignalCosts.rescaleSignalUtilityForCost(signal, mind, factorOfDeviationFromUniform)
			self.assertAlmostEqual(rescaledSignalUtiltiyOneToken, expectedResult[1])
			self.assertAlmostEqual(rescaledSignalUtiltiyOneToken_unevenCosts, expectedResult[3])
		


	def tearDown(self):
		pass
 
if __name__ == '__main__':
	unittest.main(verbosity=2)