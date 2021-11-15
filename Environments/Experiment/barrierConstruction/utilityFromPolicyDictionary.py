import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..','..', '..'))

import numpy as np

from Environments.Experiment.barrierConstruction.ValueIteration import DeterministicValueIteration
from Environments.Experiment.barrierConstruction.rewardTable import createRewardTable
from Environments.Experiment.barrierConstruction.transitionTable import createTransitionTable

class SetupPolicyTableForEnvironment(object):
	def __init__(self, stateSet, actionSet, rewardValue=10, actionCost=-1, convergenceTolerance=10e-7, gamma=.999):
		self.stateSet = stateSet
		self.actionSet = actionSet
		self.valueTable = {s:0 for s in stateSet}

		self.rewardValue = rewardValue
		self.actionCost = actionCost

		self.convergenceTolerance = convergenceTolerance
		self.gamma = gamma

	def __call__(self, barrierList):
		gridWidth = max([state[0] for state in self.stateSet])+1
		gridHeight = max([state[1] for state in self.stateSet])+1
		getTransitionTable = createTransitionTable(gridWidth, gridHeight, self.actionSet)
		transitionTable = getTransitionTable(barrierList)
		getRewardTable = createRewardTable(transitionTable, self.actionSet)
		stateRewardTables = {state: getRewardTable(self.actionCost, self.rewardValue, [state]) for state in self.stateSet}
		
		policyDictionary = {}
		valueTableDictionary = {}
		for state, rewardTable in stateRewardTables.items():
			utilityValueIteration = DeterministicValueIteration(transitionTable, rewardTable, {s:0 for s in self.stateSet}, self.convergenceTolerance, self.gamma) #create object of class value iteration --> used to obtain policy
			vt, policy = utilityValueIteration()
			print("value of state (0,3)", vt[(0,3)])
			policyDictionary[state] = policy
			valueTableDictionary[state] = vt
			print("policy generated for goal state: ", state, "\n")
		return(policyDictionary,valueTableDictionary)

# RSA action Cost - individual cost function
class SetupIndividualActionCost(object):
	def __init__(self, policyDict, transitionTable, stepCost = 1):
		self.policyDictionary = policyDict
		self.transitionTable = transitionTable
		self.stepCost = stepCost

	def __call__(self, startingPoint, endingPoint):
		policy = self.policyDictionary[endingPoint]
		currentState = startingPoint
		trajectory= [currentState]
		
		while currentState != endingPoint:
			actionDictionary = policy[currentState]
			actions = list(actionDictionary.keys())
			actionProbabilities = list(actionDictionary.values())
			actionIndicies = list(range(len(actions)))
			sampledActionIndex = np.random.choice(actionIndicies, size =1, p=actionProbabilities)[0]
			sampledAction = actions[sampledActionIndex]
			nextStateDictionary = self.transitionTable[currentState][sampledAction]
			nextState = list(nextStateDictionary.keys())[0] # if the next state is not deterministic would need to sample
			currentState = nextState
			trajectory.append(currentState)

		stepsTaken = len(trajectory) - 1
		totalCost = -abs(stepsTaken * self.stepCost)
		return(totalCost)