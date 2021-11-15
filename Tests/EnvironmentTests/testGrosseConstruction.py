import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Environments.Grosse.grosseConstruction as targetCode

@ddt
class TestGrosseFunctions(unittest.TestCase):
	def setUp(self): 
		#alpha
		self.a_Grosse = .4

		# Grosse utility cost dictionary and rewards:
		self.costDictionary = [{'L':-10, 'R':-1, 'n':0},{'L':-5, 'R':-5, 'n':0}]
		self.rewardValue = 15

		# Grosse action
		self.batteryActions = [('n', 'n'), ('L', 'n'), ('R', 'n'), ('n', 'L'), ('n', 'R'), ('L', 'R'), ('R', 'L'), ('L', 'L'), ('R', 'R')]

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
	
	#p(g|w,d) -- uniform over possible consistent goals
	# example 1 tests a target goal that is not in the world
	# example 2 tests a target goal in a constricted world (only other possibility is no goal)
	# example 3 tests a target goal in the richest world
	# example 4 tests null goal in null world (only possibility)
	@data((['n', 'L', 'R', 'LR'], 'L', 1, 'R', 0), 
		(['n', 'L', 'R', 'LR'], 'L', 1, 'L', 0.5), 
		(['n', 'L', 'R', 'LR'], 'LR', 1, 'R', .25), 
		(['n', 'L', 'R', 'LR'], 'n', 1, 'n', 1.0))
	@unpack
	def test_getGoalGivenWorldAndDesire_Grosse(self, goalList, world, desire, targetGoal, expectedResult):
		goalPDF = targetCode.getGoalGivenWorldAndDesire_Grosse(goalList, world, desire)
		self.assertEqual(goalPDF.loc[targetGoal].values[0], expectedResult)

	#helper function for p(g|w, d) Grosse
	# examples 1, 2, 3,4 test goals in full world
	# examples 5, 6 test goals in partial world
	# example 7, 8, 9 test goals in null world
	@data(('L', 'LR', 'n', 'n', 1), ('R', 'LR', 'n', 'n', 1), ('n', 'LR', 'n', 'n', 1), ('LR', 'LR', 'n', 'n', 1),
		('L', 'R', 'n', 'n', 0), ('R', 'R', 'n', 'n', 1), ('n', 'R', 'n', 'n', 1), 
		('L', 'n', 'n', 'n', 0), ('LR', 'n', 'n', 'n', 0), ('n', 'n', 'n', 'n', 1))
	@unpack
	def test_getConditionGoalPDF_Grosse(self, goal, world, nullWorld, nullGoal, expectedResult):
		conditionGoal = targetCode.getConditionGoalPDF_Grosse(goal, world)
		self.assertEqual(conditionGoal, expectedResult)
	
	####################################################################################
	@data()
	@unpack
	def test_GoalDistribution_NonUniform():
		pass
	####################################################################################

	# p(a|w, g) Grosse Battery
	# example 1, 2,3 test hard maximization + non-optimal actions
	# example 4 tests hard maximization + optimal action
	# examples 5, 6 test soft maximization + non-optimal actions
	# example 7 tests soft maximization + optimal action
	@data(('LR', 'L', ('n', 'n'), False, 0), ('LR', 'L', ('L', 'n'), False, 0), ('LR', 'L', ('R', 'n'), False, 0), 
		('LR', 'L', ('n', 'L'), False, 1.0), 
		('LR', 'L', ('n', 'n'), True, 0.009757828851297874), ('LR', 'L', ('R', 'n'), True, 0.006540868284809881), 
		('LR', 'L', ('n', 'L'), True, 0.5327594036209048))
	@unpack
	def test_ActionDistributionGivenWorldGoal_Grosse(self, world, goal, testAction, softmax, expectedResult):	
		utilityFunction = targetCode.ActionUtility_Grosse(self.costDictionary, self.rewardValue)
		getActionDistribution = targetCode.ActionDistributionGivenWorldGoal_Grosse(self.a_Grosse, utilityFunction, softmax)
		batteryActionPDF = getActionDistribution(self.batteryActions, world, goal)

		prob = batteryActionPDF.loc[testAction].values[0]
		if type(prob) != float:
			prob = prob[0]
		self.assertAlmostEqual(prob, expectedResult)

	# Action Utilities Grosse
	# examples 1, 2 test joint actions that correctly and only get the correct goal
	# examples 3, 4, 5, 6 test joint actions that do not get the goal
	# examples 7, 8, 9 test joint actions that get the goal but do too much
	# example 10 tests action where the goal isnt in the world
	@data((('L', 'n'), 'LR', 'L', 5), (('n', 'L'), 'LR', 'L', 10), 
		(('n', 'R'), 'LR', 'L', -5), (('R', 'n'), 'LR', 'L', -1), (('R', 'R'), 'LR', 'L', -6), (('n', 'n'), 'LR', 'L', 0), 
		(('R', 'L'), 'LR', 'L', 9), (('L', 'R'), 'LR', 'L', 0), (('L', 'L'), 'LR', 'L', 0),
		(('n', 'R'), 'R', 'L', -5), 
		(('n', 'R'), 'LR', 'LR', 2.5))
	@unpack
	def test_getActionUtility_Grosse(self, action, world, goal, expectedResult):
		getUtility = targetCode.ActionUtility_Grosse(self.costDictionary, self.rewardValue)
		utility = getUtility(action, world, goal)
		self.assertEqual(utility, expectedResult)

	####################################################################################
	@data()
	@unpack
	def test_SignalCost_Grosse():
		pass
	####################################################################################

	def tearDown(self):
		pass
 
if __name__ == '__main__':
	unittest.main(verbosity=2)
