import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd

import unittest
from ddt import ddt, data, unpack

import Algorithms.constantNames as NC
import Algorithms.ImaginedWe.mindConstruction as targetCode
from Environments.Grosse.grosseConstruction import *
from Environments.Misyak.misyakConstruction import *

@ddt
class TestMindComponentFunctions(unittest.TestCase):
	def setUp(self): 
		# reward and action spaces
		twoRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 2)
		oneRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 1)
		wallWorldSpace = getWorldSpace(wall = True, nBoxes = 3, nRewards = 2)
		twoAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 2)
		oneAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 1)
		
		#common ground mind dictionaries
		twoRewardCommonGround = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES:[1], NC.INTENTIONS:[1], NC.ACTIONS: twoAxeActionSpace}
		oneAxeCommonGround = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES:[1], NC.INTENTIONS:[1], NC.ACTIONS: oneAxeActionSpace}
		wallCondition = {NC.WORLDS: wallWorldSpace, NC.DESIRES:[1], NC.INTENTIONS:[1], NC.ACTIONS: twoAxeActionSpace}
		
		alpha = 2
		getActionUtility = ActionUtility(costOfLocation=0, costOfNonReward=0, valueOfReward=1)
		getActionDistribution = ActionDistributionGivenWorldGoal(alpha, getActionUtility, False)

		self.getMind = targetCode.GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
		self.mindConditionsDictionary = {'twoReward': twoRewardCommonGround, 'oneAxe': oneAxeCommonGround, 'wall': wallCondition}
		# Grosse action
		batteryActions = [('n', 'n'), ('L', 'n'), ('R', 'n'), ('n', 'L'), ('n', 'R'), ('L', 'R'), ('R', 'L'), ('L', 'L'), ('R', 'R')]


	# Examples 1, 2 test two reward world and 2 axe action space
	# Examples 3, 4, 5 test one ax (two reward) world
	# Examples 6, 7, 8, 9, 10 test wall world (all possible), 2 axe action space - no costs of actions here so equally likely to take less efficient actions

	@data(('twoReward', ((1, 1, 0),1,1,(1, 1, 0)), 1/3.0), ('twoReward', ((1, 0, 1),1,1,(1, 0, 0)), 0.0), 
		('oneAxe', ((1, 1, 0),1,1,(1, 0, 0)), 1/6.0), ('oneAxe', ((1, 1, 0),1,1,(0, 0, 0)), 0.0), ('oneAxe', ((1, 1, 0),1,1,(0, 0, 1)), .0),
		('wall', ((1, 1, 0),1,1,(1, 0, 0)), 0), ('wall', ((1, 0, 0),1,1,(1, 0, 0)), 1/24.0), ('wall', ((1, 0, 0),1,1,(1, 1, 0)), 1/24.0), ('wall', ((1, 1, 0),1,1,(1, 1, 0)), 1/8.0), ('wall', ((1, 1, 1),1,1,(1, 1, 0)), 1/24.0))
	@unpack
	def test_GenerateMind(self, mindSpaceDictionaryKey, mindTupleToTest, expectedResult):
		mindCondition = self.mindConditionsDictionary[mindSpaceDictionaryKey]
		mindPrior = self.getMind(mindCondition)
		priorTestingValue = mindPrior.loc[mindTupleToTest]
		self.assertAlmostEqual(priorTestingValue.values[0], expectedResult)

	def tearDown(self):
		pass
 
if __name__ == '__main__':
	unittest.main(verbosity=2)