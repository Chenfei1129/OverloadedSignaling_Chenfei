import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import getMultiIndexMindSpace
import Environments.Misyak.misyakConstruction as targetCode


@ddt
class TestMindComponentFunctions(unittest.TestCase):
	def setUp(self): 
		#alpha
		self.a = 5

		# Miysak utility cost/ reward examples:
		self.costReward = [0,1,0]
		self.costRewardPenalized = [(3, 2, 1), 1, .1]
		self.costRewardPenalized2 = [1, 5, 1]
		# reward and action spaces
		self.twoRewardWorldSpace = targetCode.getWorldSpace(wall = False, nBoxes = 3, nRewards = 2)
		self.twoAxeActionSpace = targetCode.getActionSpace(nBoxes = 3, nReceiverChoices = 2)
		self.oneAxeActionSpace = targetCode.getActionSpace(nBoxes = 3, nReceiverChoices = 1)
		
		#common ground mind dictionaries
		self.twoRewardCommonGround = {NC.WORLDS: self.twoRewardWorldSpace, NC.DESIRES:[1], NC.INTENTIONS:[1], NC.ACTIONS: self.twoAxeActionSpace}
		self.oneAxeCommonGround = {NC.WORLDS: self.twoRewardWorldSpace, NC.DESIRES:[1], NC.INTENTIONS:[1], NC.ACTIONS: self.oneAxeActionSpace}

	#p(w) uniform -- only depends on the number of unique items in world
	# example 1 tests one world space
	# example 2 test multiple worlds in space
	# example 3 test multiple worlds (with a duplicate) -- duplicate is not counted
	@data(([(1,0,0)], 1), ([(0,0,0), (1,0,0), (0,1,0), (0,0,1)], .25), ([(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,1,0)], .25))
	@unpack
	def test_getWorldProbabiltiy_Uniform(self, worldList, expectedResult):
		worldProbabilityDistribution = targetCode.getWorldProbabiltiy_Uniform(worldList)
		worldResult = worldProbabilityDistribution.loc[worldList[0]].values[0]
		self.assertEqual(worldResult, expectedResult)

	#p(d) uniform -- only depends on the number of unique items in desires (1 for Miysak)
	# example 1 tests one desire space
	# example 2 tests multiple desire space
	# example 3 tests multiple desire space (with a duplicate)
	@data(([1], 1), (['left', 'right', 'center', 'none'], .25), ([(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,1,0)], .25))
	@unpack
	def test_getDesireProbability_Uniform(self, desireList, expectedResult):
		desireProbabilityDistribution = targetCode.getDesireProbability_Uniform(desireList)
		desireResult = desireProbabilityDistribution.loc[desireList[0]].values[0]
		self.assertEqual(desireResult, expectedResult)
	
	# p(g|w,d) -- 1 for Miysak
	# example 1 test list input
	# example 2 tests list with different type of content
	# example 3 tests tuple
	@data(([1],(1,1,0), 1, 1), (['none'],(1,1,0), 1, 1), ([(1,1,0)],(1,1,0), 1, 1))
	@unpack
	def test_ggetGoalGivenWorldAndDesire_Uniform(self, goalList, world, desire, expectedResult):
		goalProbabilityDistribution = targetCode.getGoalGivenWorldAndDesire_Uniform(goalList, world, desire)
		goalResult = goalProbabilityDistribution.loc[goalList[0]].values[0]
		self.assertEqual(goalResult, expectedResult)
	
	# pdf of actions p(a|w,g), for a specific action of choice Misyak
	# expected rewards contain 2 values: first is the probability of a particular action in the world where the receiver has 1 ax
	# the second is the p(action) in world with 2 axes
	# examples 1, 2 test hard maximization + difference between when the utility is penalizing inefficiency v not
	# examples 3 tests hard maximization + action only getting partial rewards
	# example 4 tests hard maximization + action that does not get any rewards
	# example 5, 6 test softmax + difference between when the utility is penalizing inefficiency v not (1, 2)
	# example 7 tests softmax + action only getting partial rewards (3)
	uEfficiency = targetCode.ActionUtility(costOfLocation=0,costOfNonReward=0.1, valueOfReward=1)
	uNoPenalty = targetCode.ActionUtility(costOfLocation=0,costOfNonReward=0, valueOfReward=1)
	@data(((1,0,0), 1, (1,0,0), False, uEfficiency, [1.0, 1.0]), 
		((1,0,0), 1, (1,0,0), False, uNoPenalty, [1.0, 1.0/3]), 
		((1,1,0), 1, (1,0,0), False, uNoPenalty, [0.5, 0.0]), 
		((0,1,0), 1, (1,0,0), False, uNoPenalty, [0.0, 0.0]),
		((1,0,0), 1, (1,0,0), True, uEfficiency, [0.9853075957610499, 0.44833970720143224]), 
		((1,0,0), 1, (1,0,0), True, uNoPenalty, [0.980186662653491, 0.3303653543361982]), 
		((1,1,0), 1, (1,0,0), True, uNoPenalty, [0.4966535745378576, 0.00656053320354721]))
	@unpack
	def test_ActionDistributionGivenWorldGoal(self, world, goal, testAction, softmax, utilityFunction, expectedResult):	
		getActionDistribution = targetCode.ActionDistributionGivenWorldGoal(self.a, utilityFunction, softmax)

		oneAxeActionPDF = getActionDistribution(self.oneAxeActionSpace, world, goal)
		twoAxeActionPDF = getActionDistribution(self.twoAxeActionSpace, world, goal)

		self.assertAlmostEqual(oneAxeActionPDF.loc[testAction].values[0], expectedResult[0])
		self.assertAlmostEqual(twoAxeActionPDF.loc[testAction].values[0], expectedResult[1])

	#Action utilities - Miysak
	# expected rewards contain three values: first is the typical no penalty for incorrect choices, 
	# second location based penalty and small penalty for opening an empty box, 
	# third is similar to second but with the same penalty for going to each location
	# examples 1, 2, 3 test utility of actions corresponding exactly to the true world
	# examples 4, 5 test utility of partial (correct actions) i.e. the world has 2 rewards and the action gets one of them
	# examples 6, 7 test utility of incorrect actions 
	# example 8 tests utility of incorrect and correct actions i.e. open 2 boxes: 1 with reward 1 without
	@data(((0,0,1), (0,0,1), [1.0, 0.0, 4.0]), ((1,0,0), (1,0,0), [1.0, -2.0, 4.0]), ((1,1,0), (1,1,0), [2.0,-3.0,8.0]), 
		((1,0,0), (1,1,0), [1.0, -2.0, 4.0]), ((0,1,0), (1,1,0), [1.0,-1.0,4.0]),
		((0,1,0),(1,0,0), [0.0,-2.1,-2.0]), ((0,0,1),(1,0,0), [0,-1.1, -2]),
		((1,1,0), (1,0,0), [1.0, -4.1, 2.0]) )
	@unpack
	def test_getActionUtility(self, action, world, expectedResult):
		getActionUtility = targetCode.ActionUtility(self.costReward[0], self.costReward[1], self.costReward[2])
		utility = getActionUtility(action, world)
		self.assertEqual(utility, expectedResult[0])

		getActionUtilityPenalized  = targetCode.ActionUtility(self.costRewardPenalized[0], self.costRewardPenalized[1], self.costRewardPenalized[2])
		utilityPenalized = getActionUtilityPenalized(action, world)
		self.assertEqual(utilityPenalized, expectedResult[1])
		
		getActionUtilityPenalized2  = targetCode.ActionUtility(self.costRewardPenalized2[0], self.costRewardPenalized2[1], self.costRewardPenalized2[2])
		utilityPenalized2 = getActionUtilityPenalized2(action, world)
		self.assertEqual(utilityPenalized2, expectedResult[2])

	# examples 1, 2 test positive costs give the same results as negative ones
	# examples 3 ,4 ,5 test the range of possible signals (with same scheme as 1, 2)
	# examples 6, 7 covers integer and float costs instead of location based
	# example 8 covers non-standard markers for signal and no signal
	@data(((5,1), 1, 0, (1,0), -5), ((-5,-1), 1, 0, (1,0), -5),
	 ((-5,-1), 1, 0, (0,1), -1), ((-5,-1), 1, 0, (0,0), 0), ((-5,-1), 1, 0, (1,1), -6),
	 (1, 1, 0, (1,0), -1), (1.5, 1, 0, (1,0), -1.5),
	 ((-5,-1), 3, 2, (3,2), -5))
	@unpack
	def test_SignalCost_Misyak(self, signalCosts, signalMarker, nullMarker, signal, expectedResult):
		getSignalCost = targetCode.SignalCost_Misyak(signalCosts,signalMarker, nullMarker)
		cost = getSignalCost(signal)
		self.assertEqual(cost, expectedResult)

	# examples 1, 2, 3, 4 test output utility for entire signal distribution with same action/signal utility scheme and mind
	# example 5 tests other world with most rational action
	# examples 6, 7 test other minds where actions are unfavorable in the world
	# examples 8, 9, 10, 11 test null signals and or null actions
	@data(((-5,-1), (-1, -7), 15, 0, (1,0),{NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (1,0)}, 9), 
		((-5,-1), (-1, -7), 15, 0, (0,1), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (1,0)}, 13), 
		((-5,-1), (-1, -7), 15, 0, (0,0), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (1,0)}, 14), 
		((-5,-1), (-1, -7), 15, 0, (1,1), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (1,0)}, 8), 
		
		((-5,-1), (-1, -7), 15, 0, (0,1), {NC.WORLDS:(0,1), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (0,1)}, 7), 
		
		((-5,-1), (-1, -7), 15, 0, (1,0), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (0,1)}, -12), 
		((-5,-1), (-1, -7), 15, 0, (0,1), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (0,1)}, -8), 
		
		((-5,-1), (-1, -7), 15, 0, (0,0), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (0,0)}, 0), 
		((-5,-1), (-1, -7), 15, 0, (1,0), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (0,0)}, -5), 
		((-5,-1), (-1, -7), 15, 0, (0,0), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (1,0)}, 14), 
		((-5,-1), (-1, -7), 15, 0, (0,0), {NC.WORLDS:(1,0), NC.DESIRES:1, NC.INTENTIONS:1, NC.ACTIONS: (0,1)}, -7))
	@unpack
	def test_SignalUtility_Misyak(self, signalCosts, costOfLocation, valueOfReward, costOfNonReward, signal, mind, expectedResult):
		#generate an example multiindex condition of the mind
		mindLabels = list(mind.keys())
		mindValues = [[v] for v in mind.values()]
		idx = pd.MultiIndex.from_product(mindValues, names=mindLabels)
		mindCondition = pd.DataFrame(index=idx)

		getSignalCost = targetCode.SignalCost_Misyak(signalCosts)
		getActionUtility = targetCode.ActionUtility(costOfLocation, valueOfReward, costOfNonReward)
		getSignalUtility = targetCode.SignalUtility(getSignalCost, getActionUtility)

		utility = getSignalUtility(signal, mindCondition)
		self.assertEqual(utility, expectedResult)


	def tearDown(self):
		pass
 
if __name__ == '__main__':
	unittest.main(verbosity=2)