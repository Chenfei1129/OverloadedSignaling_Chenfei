import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack

import Algorithms.JointUtility.utilityOnlyModel as targetCode

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




# Tests for Signaler Utility
@ddt
class TestUtilityModelforSignaler(unittest.TestCase):
	def setUp(self): 
		signalerLocation = (2,1)
		receiverLocation = (2,3)
		signalSpace_CompleteMapping = ['green', 'circle']
		signalSpace_fullVocabulary = ['green', 'circle', 'purple', 'triangle']
		signalSpace_incompleteMapping = ['purple', 'circle']

		targetDictionary = {(0,1): 'purple circle', (0,2): 'green triangle', (0,3):'green circle'}

		beta = 1
		valueOfReward = 10

		self.getSignaler_fullVocab = targetCode.UtilityDrivenSignaler(
			signalSpace=signalSpace_fullVocabulary, 
			signalerLocation=signalerLocation, 
			receiverLocation=receiverLocation, 
			targetDictionary=targetDictionary, 
			valueOfReward=valueOfReward, 
			rationality=beta)

		self.getSignaler_completeMapping = targetCode.UtilityDrivenSignaler(
			signalSpace=signalSpace_CompleteMapping, 
			signalerLocation=signalerLocation, 
			receiverLocation=receiverLocation, 
			targetDictionary=targetDictionary, 
			valueOfReward=valueOfReward, 
			rationality=beta)

		self.getSignaler_incompleteMapping = targetCode.UtilityDrivenSignaler(
			signalSpace=signalSpace_incompleteMapping, 
			signalerLocation=signalerLocation, 
			receiverLocation=receiverLocation, 
			targetDictionary=targetDictionary, 
			valueOfReward=valueOfReward, 
			rationality=beta)

		self.s = signalerLocation
		self.r = receiverLocation


	@data(('green', 'green circle'), 
		('green', 'green triangle'), 
		('circle', 'green circle'))
	@unpack
	def test_intentionSignalConsistency_consistentSignalSelected(self, signal, trueGoalFeatures):
		itemConsistency = self.getSignaler_completeMapping.intentionConsistentWithSignal(signal,trueGoalFeatures)
		self.assertTrue(itemConsistency)

	@data(('green', 'purple circle'), 
		('circle', 'green triangle'), 
		('green', 'purple triangle'))
	@unpack
	def test_intentionSignalConsistency_inconsistentSignalSelected(self, signal, trueGoalFeatures):
		itemConsistency = self.getSignaler_completeMapping.intentionConsistentWithSignal(signal,trueGoalFeatures)
		self.assertFalse(itemConsistency)

	@data(((0,1), 'purple circle', 8), ((0,2), 'green triangle', 7), ((0,3), 'green circle', 6))
	@unpack
	def test_getActionUtility_signalerToTrueGoal(self, action, goal, expectedResult):
		utility = self.getSignaler_completeMapping.getActionUtility(action, goal, self.s)
		self.assertEqual(utility, expectedResult)

	@data(((0,1), 'purple circle', 6), ((0,2), 'green triangle', 7), ((0,3), 'green circle', 8))
	@unpack
	def test_getActionUtility_receiverToTrueGoal(self, action, goal, expectedResult):
		utility = self.getSignaler_completeMapping.getActionUtility(action, goal, self.r)
		self.assertEqual(utility, expectedResult)

	@data(((0,1), 'green circle', -2), ((0,2), 'green circle', -3), ((0,3), 'purple circle', -4))
	@unpack
	def test_getActionUtility_signalerMissesTrueGoal(self, action, goal, expectedResult):
		utility = self.getSignaler_completeMapping.getActionUtility(action, goal, self.s)
		self.assertEqual(utility, expectedResult)

	@data(((0,1), 'green circle', -4), ((0,2), 'green circle', -3), ((0,3), 'purple circle', -2))
	@unpack
	def test_getActionUtility_receiverMissesTrueGoal(self, action, goal, expectedResult):
		utility = self.getSignaler_completeMapping.getActionUtility(action, goal, self.r)
		self.assertEqual(utility, expectedResult)

	@data(('purple circle', 0.8807970779778825), ('green triangle', .5), ('green circle', 0.11920292202211755))
	@unpack
	def test_getAgentActionProbability(self, trueGoal, expectedResult):
		pSig, pRec = self.getSignaler_completeMapping.getAgentActionProbability(trueGoal)

		self.assertAlmostEqual(pSig, expectedResult)
		self.assertAlmostEqual(pRec, 1-expectedResult)

	@data(('green triangle', 'green', .5), ('green circle', 'green', 0.8807970779778825/2), ('green circle', 'circle', 0.8807970779778825/2))
	@unpack
	def test_getSignalerChoice_CompleteMappingSignals(self, trueGoal, signalOfInterest, expectedResult):
		signalingDistribution = self.getSignaler_completeMapping({'intentions':trueGoal})
		signalProb = signalingDistribution.loc[signalOfInterest].values[0]
		self.assertAlmostEqual(signalProb, expectedResult)

	@data(('green triangle', 'green triangle', 1.0), ('green circle', 'circle', 0.8807970779778825), ('purple circle', 'circle', 0.11920292202211755/2))
	@unpack
	def test_getSignalerChoice_IncompleteMappingSignals(self, trueGoal, signalOfInterest, expectedResult):
		signalingDistribution = self.getSignaler_incompleteMapping({'intentions':trueGoal})
		print(signalingDistribution)
		signalProb = signalingDistribution.loc[signalOfInterest].values[0]
		self.assertAlmostEqual(signalProb, expectedResult)

	@data(('green triangle', 'green', .25), ('green circle', 'circle', 0.8807970779778825/2), ('purple circle', 'circle', 0.11920292202211755/2))
	@unpack
	def test_getSignalerChoice_FullVocabulary(self, trueGoal, signalOfInterest, expectedResult):
		signalingDistribution = self.getSignaler_fullVocab({'intentions':trueGoal})
		print(signalingDistribution)
		signalProb = signalingDistribution.loc[signalOfInterest].values[0]
		self.assertAlmostEqual(signalProb, expectedResult)

	def tearDown(self):
		pass


# Tests for Self-Utility Driven Receiver
@ddt
class TestUtilityModelforReceiver(unittest.TestCase):
	def setUp(self): 
		signalerLocation = (2,1)
		receiverLocation = (2,3)

		targetDictionary = {(0,1): 'purple circle', (0,2): 'green triangle', (0,3):'green circle'}

		beta = 1
		valueOfReward = 10

		self.getReceiver = targetCode.UtilityDrivenReceiver(
			signalerLocation=signalerLocation, 
			receiverLocation=receiverLocation, 
			targetDictionary=targetDictionary, 
			valueOfReward=valueOfReward, 
			rationality=beta)

	@data(('green', 'green circle'), 
		('green', 'green triangle'), 
		('circle', 'green circle'))
	@unpack
	def test_intentionSignalConsistency_consistentSignalSelected(self, signal, trueGoalFeatures):
		itemConsistency = self.getReceiver.intentionConsistentWithSignal(signal,trueGoalFeatures)
		self.assertTrue(itemConsistency)

	@data(('green', 'purple circle'), 
		('circle', 'green triangle'), 
		('green', 'purple triangle'))
	@unpack
	def test_intentionSignalConsistency_inconsistentSignalSelected(self, signal, trueGoalFeatures):
		itemConsistency = self.getReceiver.intentionConsistentWithSignal(signal,trueGoalFeatures)
		self.assertFalse(itemConsistency)

	"""	
	@data(('circle', 1.0))
	@unpack
	def test_receiverMind_GreenTriangle(self, signal, expectedResult):
		targetPDF = self.getReceiver(signal)
		probabilityOfMind = targetPDF.loc[1]['green circle'][((2,1), (0,3))][1].values[0]
		print(probabilityOfMind)
		self.assertAlmostEqual(probabilityOfMind, expectedResult)"""



	def tearDown(self):
		pass



if __name__ == '__main__':
	unittest.main(verbosity=2)