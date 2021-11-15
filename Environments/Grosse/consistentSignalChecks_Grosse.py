import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import Algorithms.constantNames as NC

# Grosse - returns a boolean of whether the input signal is possible given the mind (action and goal for Grosse) and signal category
def signalIsConsistent_Grosse(signal, mind, signalerType, nullAction = 'n', nullSignal = 'null', receiverSignal = 'help'):
    action = mind[NC.ACTIONS]
    goal = mind[NC.INTENTIONS]

    if receiverSignal in signal:
        if 'L' in signal:
            isActionConsistent = (action[1] == 'L')
            isGoalConsistent = ('L' in goal)
        elif 'R' in signal:
            isActionConsistent = (action[1] == 'R')
            isGoalConsistent = ('R' in goal)
        else:
            #action must include the receiver (second position)
            isActionConsistent = (action[1] != nullAction)
            if 'either' in signal:
                isGoalConsistent = (goal == 'either')
            else:
                isGoalConsistent = (goal != nullAction)
    else:
        #action must NOT include the receiver
        isActionConsistent = (action[1] == nullAction)
        isGoalConsistent = True
    #isGoalConsistent = True #with this line we dont care whether the goal is consistent.
    return(all([isActionConsistent, isGoalConsistent]))