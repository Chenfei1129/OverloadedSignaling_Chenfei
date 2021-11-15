import sys
import pandas as pd
sys.path.append('../src/')
sys.path.append('../tasks/')
from GenerativeSignaler import SignalerZero
from OverloadedReceiver import ReceiverZero
from OverloadedSignaler import SignalerOne

from consistentSignalChecks import signalIsConsistent_Boxes
from commonGroundConstruction import *
from mindConstruction import *


if __name__ == "__main__":
	#parameters from commandline
	a = sys.argv[1]
	signalCategoryPrior = sys.argv[2]
	observedWorld = sys.argv[3]

	#Possible Common Grounds Setup
	#world spaces
	twoRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 2)
	oneRewardWorldSpace = getWorldSpace(wall = False, nBoxes = 3, nRewards = 1)
	wallWorldSpace = getWorldSpace(wall = True, nBoxes = 3, nRewards = 2)

	#action spaces
	oneAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 1)
	oneAxeActionSpace.remove((0,0,0))
	twoAxeActionSpace = getActionSpace(nBoxes = 3, nReceiverChoices = 2)
	twoAxeActionSpace.remove((0,0,0))

	#signal spaces
	oneTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 1)
	oneTokenSignalSpace.remove((0,0,0))
	twoTokenSignalSpace = getSignalSpace(nBoxes=3, nSignals = 2)
	twoTokenSignalSpace.remove((0,0,0))

	#condition common ground spaces
	twoTokenCondition = {'worlds': twoRewardWorldSpace, 'desires': [1], 'goals': [1], 'actions': twoAxeActionSpace}
	inversionCondition = {'worlds': twoRewardWorldSpace, 'desires': [1], 'goals': [1], 'actions': twoAxeActionSpace}
	oneAxeCondition = {'worlds': twoRewardWorldSpace, 'desires': [1], 'goals': [1], 'actions': oneAxeActionSpace}
	wallCondition = {'worlds': wallWorldSpace, 'desires': [1], 'goals': [1], 'actions': twoAxeActionSpace}

	fixedParameters = {'locationCost':0, 'rewardValue':1, 'nonRewardCost':0}

	#parameters consistent across conditions
	getActionUtility = ActionUtility(costOfLocation=0, costOfNonReward=0, valueOfReward=1)
	getActionDistribution = ActionDistributionGivenWorldGoal(a, getActionUtility, False)
	getMind = GenerateMind(getWorldProbabiltiy_Uniform, getDesireProbability_Uniform, getGoalGivenWorldAndDesire_SingleGoal, getActionDistribution)


	#Experimental Conditions
	#Two token condition
	getGenerativeSignaler_2Token = SignalerZero(twoTokenSignalSpace, signalIsConsistent_Boxes)
	getReceiverZero_2Token = ReceiverZero(commonGroundDictionary=twoTokenCondition, constructMind=getMind, getSignalerZero=getGenerativeSignaler_2Token, signalCategoryPrior=signalCategoryPrior)
	getSignalerOne_2Token = SignalerOne(alpha=a, signalSpace =twoTokenSignalSpace,  getActionUtility=getActionUtility, getReceiverZero=getReceiverZero_2Token)

	twoTokenSignalPDF = getSignalerOne_2Token({'worlds':observedWorld})

	#Inversion condition
	getGenerativeSignaler_Inversion = SignalerZero(oneTokenSignalSpace, signalIsConsistent_Boxes)
	getReceiverZero_Inversion = ReceiverZero(commonGroundDictionary=inversionCondition, constructMind=getMind, getSignalerZero=getGenerativeSignaler_Inversion, signalCategoryPrior=signalCategoryPrior)
	getSignalerOne_Inversion = SignalerOne(alpha=a, signalSpace=oneTokenSignalSpace,  getActionUtility=getActionUtility, getReceiverZero=getReceiverZero_Inversion)

	inversionSignalPDF = getSignalerOne_Inversion({'worlds':observedWorld})

	#Wall Condition
	getGenerativeSignaler_Wall = SignalerZero(oneTokenSignalSpace, signalIsConsistent_Boxes)
	getReceiverZero_Wall = ReceiverZero(commonGroundDictionary=wallCondition, constructMind=getMind, getSignalerZero=getGenerativeSignaler_Wall, signalCategoryPrior=signalCategoryPrior)
	getSignalerOne_Wall = SignalerOne(alpha=a, signalSpace=oneTokenSignalSpace,  getActionUtility=getActionUtility, getReceiverZero=getReceiverZero_Wall)

	wallSignalPDF = getSignalerOne_Wall({'worlds':observedWorld})

	#One Ax Condition
	getGenerativeSignaler_OneAxe = SignalerZero(oneTokenSignalSpace, signalIsConsistent_Boxes)
	getReceiverZero_OneAxe = ReceiverZero(commonGroundDictionary=oneAxeCondition, constructMind=getMind, getSignalerZero=getGenerativeSignaler_OneAxe, signalCategoryPrior=signalCategoryPrior)
	getSignalerOne_OneAxe = SignalerOne(alpha=a, signalSpace=oneTokenSignalSpace,  getActionUtility=getActionUtility, getReceiverZero=getReceiverZero_OneAxe)

	oneAxSignalPDF = getSignalerOne_OneAxe({'worlds':observedWorld})

	return({'twoToken': twoTokenSignalPDF, 'inversion': inversionSignalPDF, 'wall': wallSignalPDF, 'oneAx': oneAxSignalPDF})