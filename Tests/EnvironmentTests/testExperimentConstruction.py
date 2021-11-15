import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import *
from Environments.Experiment.consistentSignalChecks_Experiment import *
import Environments.Experiment.experimentConstruction as targetCode


#Tests for Euclidian, Taxicab metrics 
@ddt
class TestMetrics(unittest.TestCase):
	def setUp(self): 
		pass

	# EUCLIDIAN COSTS #######################################################################################################
	@data(((3,0), (1,0), -2), 
		((3,0), (3,8), -8))
	@unpack
	def test_calculateLocationCostEuclidianMetric_OneDimensionalMove(self, agentLocation, proposedActionLocation, expectedResult):
		locationCostEuclidian = targetCode.calculateLocationCost_EuclidianMetric(agentLocation, proposedActionLocation)
		self.assertAlmostEqual(locationCostEuclidian, expectedResult)

	@data(((3,0), (0,11),-11.40175425099138), 
		((3,0), (6,9), -9.486832980505138), 
		((3,5), (1,2), -3.605551275463989), 
		((3,5), (7,2), -5))
	@unpack
	def test_calculateLocationCostEuclidianMetric_TwoDimensionalMove(self, agentLocation, proposedActionLocation, expectedResult):
		locationCostEuclidian = targetCode.calculateLocationCost_EuclidianMetric(agentLocation, proposedActionLocation)
		self.assertAlmostEqual(locationCostEuclidian, expectedResult)
	
	@data(((3,0), (3,0), -0), 
		((3,8), (3,8), 0))
	@unpack
	def test_calculateLocationCostEuclidianMetric_NoMove(self, agentLocation, proposedActionLocation, expectedResult):
		locationCostEuclidian = targetCode.calculateLocationCost_EuclidianMetric(agentLocation, proposedActionLocation)
		self.assertAlmostEqual(locationCostEuclidian, expectedResult)

	
	# TAXICAB COSTS #######################################################################################################
	@data(((3,0), (1,0), -2), 
		((3,0), (3,8), -8))
	@unpack
	def test_calculateLocationCostTaxicabMetric_OneDimensionalMove(self, agentLocation, proposedActionLocation, expectedResult):
		locationCostTaxicab = targetCode.calculateLocationCost_TaxicabMetric(agentLocation, proposedActionLocation)
		self.assertEqual(locationCostTaxicab, expectedResult)

	@data(((3,0), (0,11),-14), 
		((3,0), (6,9), -12), 
		((3,5), (1,2), -5), 
		((3,5), (7,2), -7))
	@unpack
	def test_calculateLocationCostTaxicabMetric_TwoDimensionalMove(self, agentLocation, proposedActionLocation, expectedResult):
		locationCostTaxicab = targetCode.calculateLocationCost_TaxicabMetric(agentLocation, proposedActionLocation)
		self.assertEqual(locationCostTaxicab, expectedResult)

	@data(((3,0), (3,0), -0), 
		((3,8), (3,8), 0))
	@unpack
	def test_calculateLocationCostTaxicabMetric_NoMove(self, agentLocation, proposedActionLocation, expectedResult):
		locationCostTaxicab = targetCode.calculateLocationCost_TaxicabMetric(agentLocation, proposedActionLocation)
		self.assertEqual(locationCostTaxicab, expectedResult)

	def tearDown(self):
		pass



@ddt
class TestExperimentSetup(unittest.TestCase):
	def setUp(self):
		getCost = targetCode.calculateLocationCost_TaxicabMetric
		rewardValue = 20
		signalerLocation = (5,0)
		receiverLocation = (5,10)
		self.signalDictionary = {(3,0): 'green', (7, 0): 'circle', (5,2): 'blue'}
		self.targetDictionary = {(1,10): 'green triangle', (5,6): 'green circle', (9,10):'purple circle'}
		
		self.getJointUtility = targetCode.JointActionUtility(
			costFunction=getCost, 
			valueOfReward=rewardValue, 
			signalerLocation=signalerLocation, 
			receiverLocation=receiverLocation, 
			targetDictionary=self.targetDictionary)


		alpha = .9
		self.getActionDistribution = targetCode.ActionDistributionGivenWorldGoal(alpha = alpha, actionUtilityFunction = self.getJointUtility)

	###########################################################################################################
	##### test construction of action space
	###########################################################################################################
	@data(((5,0), (5,10), ((3,0), (9,10)) ),
		((5,0), (5,10), ((3,0), (1,10)) ),
		((5,0), (5,10), ((5,2), (9,10)) ))
	@unpack
	def test_getActionSpace_JointActionInSpace_SignalAction(self, signalerLocation, receiverLocation, actionToTest):
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionInActionSpace = actionToTest in actionSpace
		self.assertTrue(actionInActionSpace)

	@data(((5,0), (5,10), ((5,0), (5,10)) ),
		((5,0), (5,10), ((5,0), (1,10)) ),
		((5,0), (5,10), ((5,2), (5,10)) ),
		((1,2), (3,10), ((1,2), (3,10)) ),
		((1,5), (5,10), ((1,5), (1,10)) ))
	@unpack
	def test_getActionSpace_JointActionInSpace_DoNothing(self, signalerLocation, receiverLocation, actionToTest):
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionInActionSpace = actionToTest in actionSpace
		self.assertTrue(actionInActionSpace)

	@data(((5,0), (5,10), ((1,10), (1,10)) ),
		((5,0), (5,10), ((5,6), (5,6)) ))
	@unpack
	def test_getActionSpace_JointActionNotInSpace_BothAgentsToSameTarget(self, signalerLocation, receiverLocation, actionToTest):
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionInActionSpace = actionToTest in actionSpace
		self.assertFalse(actionInActionSpace)

	@data(((5,0), (5,10), ((3,0), (5,2)) ),
		((5,0), (5,10), ((5,0), (3,0)) ))
	@unpack
	def test_getActionSpace_JointActionNotInSpace_ReceiverToSignal(self, signalerLocation, receiverLocation, actionToTest):
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionInActionSpace = actionToTest in actionSpace
		self.assertFalse(actionInActionSpace)

	@data(((5,0), (5,10), ((1,1), (1,10)) ),
		((5,0), (5,10), ((3,0), (8,8)) ))
	@unpack
	def test_getActionSpace_JointActionNotInSpace_MoveToWhitespaceTile(self, signalerLocation, receiverLocation, actionToTest):
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionInActionSpace = actionToTest in actionSpace
		self.assertFalse(actionInActionSpace)


	@data(((5,0), (5,10), ((5,0), (9,10)) ),
	((5,0), (5,10), ((5,0), (1,10)) ))
	@unpack
	def test_getActionSpace_JointActionNotInSpace_SignalerMustAct(self, signalerLocation, receiverLocation, actionToTest):
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation, False)
		actionInActionSpace = actionToTest in actionSpace
		self.assertFalse(actionInActionSpace)


	###########################################################################################################
	##### test get agent utility
	###########################################################################################################
	@data(((5,0), (0,0), 'green triangle', -5), 
		((5,5), (3,4), 'green triangle', -3), 
		((5,0), (9,9), 'green triangle', -13), 
		((5,0), (3,0), 'green triangle', -2))
	@unpack
	def test_getUtility_emptySquare(self, agentPosition, action, goal, expectedResult):
		utility = self.getJointUtility.getUtility(agentPosition, action, goal)
		self.assertEqual(utility, expectedResult)

	@data(((5,0), (3,0), 'green triangle', -2), 
		((5,0), (7,0), 'green triangle', -2), 
		((5,0), (5,2), 'green triangle', -2))
	@unpack
	def test_getUtility_signal(self, agentPosition, action, goal, expectedResult):
		utility = self.getJointUtility.getUtility(agentPosition, action, goal)
		self.assertEqual(utility, expectedResult)

	@data(((5,0), (5,6), 'green triangle', -6), ((5,10), (9,10), 'green triangle', -4))
	@unpack
	def test_getUtility_incorrectGoalLocation(self, agentPosition, action, goal, expectedResult):
		utility = self.getJointUtility.getUtility(agentPosition, action, goal)
		self.assertEqual(utility, expectedResult)

	@data(((5,0), (1,10), 'green triangle', 6), ((5,10), (5,6), 'green circle', 16), ((5,10), (9,10), 'purple circle', 16))
	@unpack
	def test_getUtility_CorrectGoalLocation(self, agentPosition, action, goal, expectedResult):
		utility = self.getJointUtility.getUtility(agentPosition, action, goal)
		self.assertEqual(utility, expectedResult)

	###########################################################################################################
	##### test action distribution
	###########################################################################################################
	@data((((1,10), (5,10)), 'green triangle'),  (((9,10), (5,10)), 'purple circle'))
	@unpack
	def test_actionDistribution_ImprobableActionReachesGoal(self, actionToTest, goal):
		signalerLocation = (5,0)
		receiverLocation = (5,10)
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionDistribution = self.getActionDistribution(actionSpace, [1], goal)

		actionProbability = actionDistribution.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(actionToTest)]
		epsilonThreshold = .001
		self.assertTrue(actionProbability.values[0] < epsilonThreshold)

	@data((((3,0), (5,6)), 'green triangle'), (((5,2), (5,10)), 'green triangle'), (((7,0), (9,10)), 'green triangle'))
	@unpack
	def test_actionDistribution_ImprobableActionMissesGoal(self, actionToTest, goal):
		signalerLocation = (5,0)
		receiverLocation = (5,10)
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionDistribution = self.getActionDistribution(actionSpace, [1], goal)

		actionProbability = actionDistribution.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(actionToTest)]
		epsilonThreshold = .001
		self.assertTrue(actionProbability.values[0] < epsilonThreshold)


	@data((((3,0), (1,10)), ((7,0), (1,10)), 'green triangle'))
	@unpack
	def test_actionDistribution_JointActionsWithEqualUtilities(self, firstAction, secondAction, goal):
		signalerLocation = (5,0)
		receiverLocation = (5,10)
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionDistribution = self.getActionDistribution(actionSpace, [1], goal)

		action1Probability = actionDistribution.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(firstAction)].values[0]
		action2Probability = actionDistribution.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(secondAction)].values[0]
		self.assertAlmostEqual(action1Probability, action2Probability)


	@data((((5,0), (1,10)), ((7,0), (1,10)), 'green triangle'), 
		(((5,0), (1,10)), ((5,0), (5,10)), 'green triangle'), 
		(((5,0), (1,10)), ((3,0), (1,10)), 'green triangle') )
	@unpack
	def test_actionDistribution_JointActionsWithUnequalUtilities(self, firstAction, secondAction, goal):
		signalerLocation = (5,0)
		receiverLocation = (5,10)
		actionSpace = targetCode.getActionSpace(self.targetDictionary, self.signalDictionary, signalerLocation, receiverLocation)
		actionDistribution = self.getActionDistribution(actionSpace, [1], goal)

		action1Probability = actionDistribution.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(firstAction)].values[0]
		action2Probability = actionDistribution.reset_index().astype({NC.ACTIONS:str}).set_index(NC.ACTIONS).loc[str(secondAction)].values[0]
		self.assertTrue(action1Probability > action2Probability)



	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest.main(verbosity=2)