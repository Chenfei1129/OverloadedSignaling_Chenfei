"""
Created on Mon Jun 29 22:54:16 2020

@author: adelphachan

Joint Utility Obstacles MDP Transition Table
"""
import numpy as np

"""
Create transition table

Constructor:
    state set (grid dimensions)
    action set (list of actions as tuples)
    
Callable:
        list of transition barriers, list of tuples
            {(state, action)}

Output: 
    nested dictionary 
        {state:{action:{nextState:probability}}}

"""

class createTransitionTable(object):
  
  #constructor
    def __init__(self, gridWidth, gridHeight, actionSet):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.stateSet = [(i,j) for i in range(gridWidth) for j in range(gridHeight)]
        self.actionSet = actionSet
      
  #callable: output = list ONLY with barriers
    def __call__(self, transitionTableBarriers): #input = list of barriers
        transitionTable = {state: self.getStateTransitionTable(state) for state in self.stateSet} #set up initial transitionTable
        
        for pair in transitionTableBarriers:
            state = pair[0]
            action = pair[1]
            transitionTable[state][action] = {state:1}
        
        return(transitionTable)
      

  #helper functions for callable
    def getStateTransitionTable(self, state):
        actionTransitionDistribution = {action: self.getStateActionTransitionTable(state,action) for action in self.actionSet}
        return(actionTransitionDistribution)
  
    def getStateActionTransitionTable(self, currentState, action):
        nextState = self.getNextState(currentState, action)
        transitionDistribution = {nextState: 1}
        return(transitionDistribution)
  
    def getNextState(self, state, action):

        xCoord = state[0] + action[0]
        yCoord = state[1] + action[1]
     
        potentialNextState = (xCoord,yCoord)
     
        if potentialNextState in self.stateSet:
            state = potentialNextState
        return(state)
 
