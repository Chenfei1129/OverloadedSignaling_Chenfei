import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import GenerateMind
from Algorithms.ImaginedWe.GenerativeSignaler import SignalerZero
import Algorithms.ImaginedWe.OverloadedReceiver as targetCode
from Environments.Misyak.consistentSignalChecks_Misyak import signalIsConsistent_Boxes
from Environments.Misyak.misyakConstruction import *
from Environments.Grosse.consistentSignalChecks_Grosse import signalIsConsistent_Grosse
from Environments.Grosse.grosseConstruction import *

@ddt
class TestOverloadedReceiver(unittest.TestCase):
	def setUp(self): 
		#world spaces
		twoRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 2)
		oneRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 1)
		wallWorldSpace = getWorldSpace(wall = True, nBoxes = 3, nRewards = 2)

		#action spaces
		oneAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 1)
		twoAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 2)

		#signal spaces
		oneTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 1)
		twoTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 2)

		#condition common ground spaces
		oneRewardCommonGround = {NC.WORLDS: oneRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: oneAxeActionSpace}
		twoRewardCommonGround = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: twoAxeActionSpace}

		oneAxeCommonGround = {NC.WORLDS: twoRewardWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: oneAxeActionSpace}
		wallCommonGround = {NC.WORLDS: wallWorldSpace, NC.DESIRES: [1], NC.INTENTIONS: [1], NC.ACTIONS: twoAxeActionSpace}
		
		#action utility
		a = 5
		getActionUtility = ActionUtility(costOfLocation=0, costOfNonReward=0, valueOfReward=1)

		#action distribution
		getActionDistribution = ActionDistributionGivenWorldGoal(a, getActionUtility, False)
		getActionDistributionSoftmax = ActionDistributionGivenWorldGoal(a, getActionUtility, True)	

		#minds	
		getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistribution)
		getMindSoftmax = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_Uniform, getActionDistributionSoftmax)		

		#signaler type prior
		signalCategoryPrior_Uniform = {'1':.5, '-1':.5}
		signalCategoryPrior_Biased = {'1':.65, '-1':.45}

		#signaler zero 
		getSignaler0_TwoToken = SignalerZero(twoTokenSignalSpace, signalIsConsistent_Boxes)
		getSignaler0_OneToken = SignalerZero(oneTokenSignalSpace, signalIsConsistent_Boxes)

		#receiver zero conditions
		self.getReceiver0_OneReward = targetCode.ReceiverZero(commonGroundDictionary=oneRewardCommonGround, constructMind=getMind, getSignalerZero=getSignaler0_OneToken, signalCategoryPrior=signalCategoryPrior_Uniform)
		self.getReceiver0_TwoToken = targetCode.ReceiverZero(commonGroundDictionary=twoRewardCommonGround, constructMind=getMind, getSignalerZero=getSignaler0_TwoToken, signalCategoryPrior=signalCategoryPrior_Uniform)
		self.getReceiver0_Inversion = targetCode.ReceiverZero(commonGroundDictionary=twoRewardCommonGround, constructMind=getMind, getSignalerZero=getSignaler0_OneToken, signalCategoryPrior=signalCategoryPrior_Uniform)
		self.getReceiver0_OneAxe = targetCode.ReceiverZero(commonGroundDictionary=oneAxeCommonGround, constructMind=getMind, getSignalerZero=getSignaler0_OneToken, signalCategoryPrior=signalCategoryPrior_Uniform)
		self.getReceiver0_Wall = targetCode.ReceiverZero(commonGroundDictionary=wallCommonGround, constructMind=getMind, getSignalerZero=getSignaler0_OneToken, signalCategoryPrior=signalCategoryPrior_Uniform)

		#inputs for testing:
		self.twoRewardMind = getMind(twoRewardCommonGround)
		self.oneRewardMind = getMind(oneRewardCommonGround)

		#GROSSE
		#utility
		costD = [{'L':-4, 'R':-1, 'n':0},{'L':-2, 'R':-2, 'n':0}]
		rewardValue = 10
		getUtilityGrosse = ActionUtility_Grosse(costD, rewardValue)
		#p(a|w,g) function
		a = 1
		getActionPDF = ActionDistributionGivenWorldGoal_Grosse(alpha = a, actionUtilityFunction=getUtilityGrosse, softmax=True)
		getGoalPDF = GoalDistribution_NonUniform({'L':.5, 'R':.5})
		#p(mind) function
		getMindGrosse = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalPDF, getActionPDF)
		#condition parameters
		handsFreeCondition = {'worlds': ['LR'], 'desires': [1], 'intentions': ['L', 'R'], 'actions': [ ('L', 'n'), ('R', 'n'), ('n', 'L'), ('n', 'R')]}
		signalCategoryPriorG = {'1':1}
		signalSpaceG = ['help', 'null']

		#inference
		getGenerativeSignalerGrosse = SignalerZero(signalSpaceG, signalIsConsistent_Grosse)
		self.getReceiverZeroGrosse = targetCode.ReceiverZero(commonGroundDictionary=handsFreeCondition, constructMind=getMindGrosse, getSignalerZero=getGenerativeSignalerGrosse, signalCategoryPrior=signalCategoryPriorG)
			
	#cases: inconsistent, positive signal, negative signal type - zero is in the signal set
	@data((('1',(0, 0, 1),1,1,(0, 0, 1),(1, 0, 0)), 0), 
		(('1',(0, 0, 1),1,1,(0, 0, 1),(0, 0, 1)), 0.5), 
		(('-1',(0, 0, 1),1,1,(0, 0, 1),(1, 0, 0)), 1.0/3) )
	@unpack
	def test_constructLikelihoodDataFrameFromMindConditions_oneReward(self, mindTuple, expectedResult):
		oneRewardLikelihood = self.getReceiver0_OneReward.constructLikelihoodDataFrameFromMindConditions(self.oneRewardMind)
		oneRewardLikelihoodTestingValue = oneRewardLikelihood.loc[mindTuple]
		self.assertEqual(oneRewardLikelihoodTestingValue.values[0], expectedResult)

	#cases: inconsistent, positive signal, negative signal type - zero is in the signal set
	@data((('1',(1, 1, 0),1,1,(1, 0, 0),(1, 0, 0)), 0.25), 
		(('1',(1, 1, 0),1,1,(1, 1, 0),(1, 0, 0)), 0.25), 
		(('-1',(1, 1, 0),1,1,(1, 1, 0),(0, 0, 1)), 0.5))
	@unpack
	def test_constructLikelihoodDataFrameFromMindConditions_twoReward(self, mindTuple, expectedResult):
		twoRewardLikelihood = self.getReceiver0_TwoToken.constructLikelihoodDataFrameFromMindConditions(self.twoRewardMind)
		twoRewardLikelihoodTestingValue = twoRewardLikelihood.loc[mindTuple]
		self.assertEqual(twoRewardLikelihoodTestingValue.values[0], expectedResult)

	@data(( {'1':.5, '-1':.5}, ('1',(1, 0, 0),1,1,(1, 0, 0)), .5/3), ( {'1':.6, '-1':.4}, ('1',(1, 0, 0),1,1,(1, 0, 0)), .6/3), 
		( {'1':.5, '-1':.5}, ('1',(1, 0, 0),1,1,(0, 1, 0)), 0.0), 
		( {'1':.6, '-1':.4}, ('-1',(1, 0, 0),1,1,(1, 0, 0)), .4/3))
	#issue with data type :(
	@unpack
	def test_constructJointMindSignalCategoryPrior(self, categoryPrior, mindTuple, expectedResult):
		twoRewardPrior = self.getReceiver0_TwoToken.constructJointMindSignalCategoryPrior(self.twoRewardMind, categoryPrior)
		twoRewardPriorTestingValue = twoRewardPrior.loc[mindTuple]
		oneRewardPrior = self.getReceiver0_OneReward.constructJointMindSignalCategoryPrior(self.oneRewardMind, categoryPrior)
		oneRewardPriorTestingValue = oneRewardPrior.loc[mindTuple]
		a = round(float(oneRewardPriorTestingValue.values), 7)
		expected = round(expectedResult, 7)
		self.assertEqual(a, expected)


	@data(('help', ('LR', 1, 'L', ('n', 'L')), .766026), ('null', ('LR', 1, 'L', ('n', 'L')), 0.0), 
		('help', ('LR', 1, 'R', ('n', 'L')), .000011), ('null', ('LR', 1, 'R', ('n', 'L')), 0.0), 
		('help', ('LR', 1, 'L', ('L', 'n')), 0.0), ('null', ('LR', 1, 'L', ('L', 'n')), .140161), 
		('help', ('LR', 1, 'R', ('L', 'n')), 0.0), ('null', ('LR', 1, 'R', ('L', 'n')), 0.000002))
	@unpack
	def test_getMindPosterior(self, signal, mindTuple, expectedResult):
		mindPosterior = self.getReceiverZeroGrosse(signal)
		mindPosteriorTestingValue = round(mindPosterior.loc[mindTuple].values[0],6)
		self.assertEqual(mindPosteriorTestingValue, expectedResult)
		
	def tearDown(self):
		pass


 
if __name__ == '__main__':
	unittest.main(verbosity=2)