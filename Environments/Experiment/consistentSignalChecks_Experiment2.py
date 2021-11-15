import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import Algorithms.constantNames as NC

class SignalIsConsistent_Experiment(object):
    def __init__(self, signalDictionary, targetDictionary, signalerLocation, receiverLocation, signalForInPlace = 'null'):
        self.targetDictionary = targetDictionary
        self.objectLocations = {**signalDictionary, **targetDictionary}
        self.signalerLocation =signalerLocation
        self.receiverLocation = receiverLocation
        self.signalForInPlace = signalForInPlace

    def __call__(self, signal, mind, signalType=None):
        action = mind[NC.ACTIONS]
        goal = mind[NC.INTENTIONS]

        if signal == self.signalForInPlace:
            return(action[0] == self.signalerLocation)
        else:
            signalGoalConsistency = self.isSignalConsistentWithGoal(signal, goal)
            signalActionConsistency = self.isSignalerActionConsistent(signal, action)
            receiverActionConsistency = self.isReceiverActionConsistent(signal, action, goal)
            return(all([signalGoalConsistency, signalActionConsistency, receiverActionConsistency]))

    """
    def signalerIsIrrational(self, signal):
        # irrational signal or action from signaler
        possibleGoalSpace = list(self.targetDictionary.values())
        consistentTargets = [g for g in possibleGoalSpace if self.isSignalConsistentWithGoal(signal, g)]
        return(len(consistentTargets) == 0)
    """

    def isSignalerActionConsistent(self, signal, action):
        signalerAction = action[0]
        # if signalerAction == self.signalerLocation:
        #    return(True)
        #else:
        signalLocation = [location for location, signalFeatures in self.objectLocations.items() if signal == signalFeatures][0]
        actionConsistentWithSignal = (signalerAction == signalLocation)
        return(actionConsistentWithSignal)

    def isReceiverActionConsistent(self, signal, action, goal):
        receiverAction = action[1]
        #Staying in place is always consistent
        if receiverAction == self.receiverLocation:
            return(True)
        # If the signaler reaches the goal, the receiver MUST stay in place
        elif signal == goal:
            return(self.receiverLocation == receiverAction)
        else:
            receiverActionString = [itemFeatures for itemLoc, itemFeatures in self.targetDictionary.items() if itemLoc == receiverAction][0]
            featuresOfSignal = list(signal.split())
            featuresOfReceiverAction= list(receiverActionString.split())
            consistentReceiverAction = [feature in featuresOfReceiverAction for feature in featuresOfSignal]
            return(all(consistentReceiverAction))
        
    def isSignalConsistentWithGoal(self, signal, goal):
        featuresOfSignal = list(signal.split())
        featuresOfGoal = list(goal.split())
        signalConsistentWithGoal = [feature in featuresOfGoal for feature in featuresOfSignal]
        return(all(signalConsistentWithGoal))


class SignalIsConsistent_ExperimentSeparatedSignals(object):
    def __init__(self, targetDictionary, signalerLocation, receiverLocation, signalForInPlace = 'null'):
        self.targetDictionary = targetDictionary
        self.signalerLocation = signalerLocation
        self.receiverLocation = receiverLocation
        self.signalForInPlace = signalForInPlace

    def __call__(self, signal, mind, signalType=None):
        action = mind[NC.ACTIONS]
        goal = mind[NC.INTENTIONS]
        #print(goal)

        if signal == self.signalForInPlace:
            return(action[0] == self.signalerLocation)
        else:
            signalGoalConsistency = self.isSignalConsistentWithGoal(signal, goal)
            signalActionConsistency = self.isSignalerActionConsistent(signal, action)
            receiverActionConsistency = self.isReceiverActionConsistent(signal, action, goal)
            return(all([signalGoalConsistency, signalActionConsistency, receiverActionConsistency]))


    def isSignalerActionConsistent(self, signal, action):
        signalerAction = action[0]
        return(signalerAction == self.signalerLocation)

    def isReceiverActionConsistent(self, signal, action, goal):
        receiverAction = action[1]
        #Staying in place is always consistent
        if receiverAction == self.receiverLocation:
            return(True)
        # If the signaler reaches the goal, the receiver MUST stay in place
        #elif signal == goal:
         #   print(self.receiverLocation == receiverAction)
          #  return(self.receiverLocation == receiverAction)
        else:
            receiverActionString = [itemFeatures for itemLoc, itemFeatures in self.targetDictionary.items() if itemLoc == receiverAction][0]
            featuresOfSignal = list(signal.split())
            featuresOfReceiverAction= list(receiverActionString.split())
            consistentReceiverAction = [feature in featuresOfReceiverAction for feature in featuresOfSignal]
            #print(featuresOfReceiverAction)
            #print(consistentReceiverAction)
            return(all(consistentReceiverAction))
        
    def isSignalConsistentWithGoal(self, signal, goal):
        featuresOfSignal = list(signal.split())
        featuresOfGoal = list(goal.split())
        
        signalConsistentWithGoal = [feature in featuresOfGoal for feature in featuresOfSignal]

        return(all(signalConsistentWithGoal))


