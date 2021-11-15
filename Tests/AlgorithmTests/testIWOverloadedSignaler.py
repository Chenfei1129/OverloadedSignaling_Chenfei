import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack

from Environments.Misyak.consistentSignalChecks_Misyak import signalIsConsistent_Boxes 
from Environments.Misyak.misyakConstruction import *
from Environments.Grosse.consistentSignalChecks_Grosse import signalIsConsistent_Grosse
from Environments.Grosse.grosseConstruction import *
import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import GenerateMind
from Algorithms.ImaginedWe.OverloadedReceiver import ReceiverZero
from Algorithms.ImaginedWe.GenerativeSignaler import SignalerZero
import Algorithms.ImaginedWe.OverloadedSignaler as targetCode

@ddt
class TestOverloadedSignaling(unittest.TestCase):
	def setUp(self): 
		twoRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 2)
		twoAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 2)

		#condition common ground spaces
		twoRewardCommonGround = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: twoAxeActionSpace}

		#action utility
		a = 5
		actionUtilityFunction = ActionUtility(costOfLocation=0, valueOfReward=1, costOfNonReward=0)
		actionUtilityFunction_LocationPenalized = ActionUtility(costOfLocation=(0.1, 0.2, 0.3), valueOfReward=1, costOfNonReward=0)
		actionUtilityFunction_NonrewardPenalized = ActionUtility(costOfLocation=0, valueOfReward=1, costOfNonReward=0.5)


		#action distribution
		getActionDistribution = ActionDistributionGivenWorldGoal(a, actionUtilityFunction, False)
		getActionDistributionSoftmax = ActionDistributionGivenWorldGoal(a, actionUtilityFunction, True)	

		getActionDistribution_LocationPenalized = ActionDistributionGivenWorldGoal(a,actionUtilityFunction_LocationPenalized, False)
		getActionDistribution_NonrewardPenalized = ActionDistributionGivenWorldGoal(a,actionUtilityFunction_NonrewardPenalized, False)

		#minds	
		getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
		getMindSoftmax = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistributionSoftmax)	

		locationPenalizedMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution_LocationPenalized)
		nonrewardPenalizedMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution_NonrewardPenalized)

		#signaler type prior
		signalCategoryPrior = {'1':.5, '-1':.5}
		twoTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 2)

		#signaler zero 
		signaler0 = SignalerZero(twoTokenSignalSpace, signalIsConsistent_Boxes)

		#receiver zero conditions
		receiver0 = ReceiverZero(commonGroundDictionary=twoRewardCommonGround, constructMind=getMind, getSignalerZero=signaler0, signalCategoryPrior=signalCategoryPrior)
		receiver0_softmax = ReceiverZero(commonGroundDictionary=twoRewardCommonGround, constructMind=getMindSoftmax, getSignalerZero=signaler0, signalCategoryPrior=signalCategoryPrior)
		receiver0_location = ReceiverZero(commonGroundDictionary=twoRewardCommonGround, constructMind=locationPenalizedMind, getSignalerZero=signaler0, signalCategoryPrior=signalCategoryPrior)
		receiver0_nonreward = ReceiverZero(commonGroundDictionary=twoRewardCommonGround, constructMind=nonrewardPenalizedMind, getSignalerZero=signaler0, signalCategoryPrior=signalCategoryPrior)

		#signaler one 
		self.signaler1 = targetCode.SignalerOne(alpha=a, signalSpace=twoTokenSignalSpace, getActionUtility = actionUtilityFunction, getReceiverZero=receiver0)
		self.signaler1_softmax = targetCode.SignalerOne(alpha=a, signalSpace=twoTokenSignalSpace, getActionUtility = actionUtilityFunction, getReceiverZero=receiver0_softmax)
		self.signaler1_location = targetCode.SignalerOne(alpha=a, signalSpace=twoTokenSignalSpace, getActionUtility = actionUtilityFunction_LocationPenalized, getReceiverZero=receiver0_location)
		self.signaler1_nonreward = targetCode.SignalerOne(alpha=a, signalSpace=twoTokenSignalSpace, getActionUtility = actionUtilityFunction_NonrewardPenalized, getReceiverZero=receiver0_nonreward)

	@data(({NC.WORLDS:(1,1,0)}, (1,1,0), [2, 1.9735810490857661, 1.7, 2]), 
		({NC.WORLDS:(1,1,0)}, (1,0,0), [1.25, 1.2433621088776834, 0.825, 0.875]), 
		({NC.WORLDS:(1,1,0)}, (0,0,0), [4.0/3, 1.3244975466785818, 0.9333333333333333, 1])) 
	@unpack
	def test_getUtilityOfSignal(self, observation, signal, expectedResult):
		utility = self.signaler1.getUtilityofSignal(observation, signal)
		self.assertAlmostEqual(utility, expectedResult[0])

		utility_softmax = self.signaler1_softmax.getUtilityofSignal(observation, signal)
		self.assertAlmostEqual(utility_softmax, expectedResult[1])
		
		utility_location = self.signaler1_location.getUtilityofSignal(observation, signal)
		self.assertAlmostEqual(utility_location, expectedResult[2])
	
		utility_nonreward = self.signaler1_nonreward.getUtilityofSignal(observation, signal)
		self.assertAlmostEqual(utility_nonreward, expectedResult[3])

	def tearDown(self):
			pass


if __name__ == '__main__':
	unittest.main(verbosity=2)