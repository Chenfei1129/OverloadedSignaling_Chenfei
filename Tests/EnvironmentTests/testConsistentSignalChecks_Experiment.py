import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import Algorithms.constantNames as NC
from Algorithms.ImaginedWe.mindConstruction import *
import Environments.Experiment.consistentSignalChecks_Experiment as targetCode


###########################################################################################################
##### Signal is Consistent - Experiment
###########################################################################################################	
@ddt
class TestSignalIsConsistentExperiment(unittest.TestCase):
	def setUp(self):
		self.signalerLocation = (5,0)
		self.receiverLocation = (5,10)

		self.signalDictionary_irrationalSignal = {(3,0): 'green', (7, 0): 'circle', (5,2): 'blue'}
		self.signalDictionary_unambiguousSignal = {(3,0): 'green', (7,0): 'triangle'}
		self.signalDictionary_bothSignalsAmbiguous= {(3,0): 'green', (7, 0): 'circle'}

		self.targetDictionary = {(1,10): 'green triangle', (5,6): 'green circle', (9,10):'purple circle'}


	@data(('null', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((5,0),(5,10))}, '1'), 
		('null',{NC.INTENTIONS: 'green triangle', NC.ACTIONS:((5,0),(1,10))}, '1'), 
		('null', {NC.INTENTIONS: 'green square', NC.ACTIONS:((5,0),(5,6))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_StayInPlace_Consistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertTrue(consistencyBoolean)

	@data(('null', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((3,0),(5,10))}, '1'), 
		('null',{NC.INTENTIONS: 'green triangle', NC.ACTIONS:((3,0),(1,10))}, '1'), 
		('null', {NC.INTENTIONS: 'green square', NC.ACTIONS:((7,0),(5,6))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_StayInPlace_Inconsistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertFalse(consistencyBoolean)

	@data(('green', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((3,0),(1,10))}, '1'), 
		('green',{NC.INTENTIONS: 'green circle', NC.ACTIONS:((3,0),(5,6))}, '1'), 
		('green', {NC.INTENTIONS: 'green triangle', NC.ACTIONS:((3,0),(5,6))}, '1'),
		('green', {NC.INTENTIONS: 'green triangle', NC.ACTIONS:((3,0),(1,10))}, '1'),
		('triangle', {NC.INTENTIONS: 'green triangle', NC.ACTIONS:((7,0),(1,10))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_SignalGoalJointAction_Consistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertTrue(consistencyBoolean)

	@data(('green circle', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((5,6),(5,10))}, '1'), 
		('green triangle',{NC.INTENTIONS: 'green triangle', NC.ACTIONS:((1,10),(5,10))}, '1'), 
		('purple circle', {NC.INTENTIONS: 'purple circle', NC.ACTIONS:((9,10),(5,10))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_SignalerGetsGoal_Consistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertTrue(consistencyBoolean)

	@data(('green circle', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((5,6),(1,10))}, '1'), 
		('green triangle',{NC.INTENTIONS: 'green triangle', NC.ACTIONS:((1,10),(5,6))}, '1'), 
		('purple circle', {NC.INTENTIONS: 'purple circle', NC.ACTIONS:((9,10),(5,6))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_SignalerGetsGoal_ReceiverActionInconsistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertFalse(consistencyBoolean)


	@data(('green', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((3,0),(9,10))}, '1'), 
		('triangle',{NC.INTENTIONS: 'green triangle', NC.ACTIONS:((7,0),(5,6))}, '1'), 
		('triangle', {NC.INTENTIONS: 'green triangle', NC.ACTIONS:((7,0),(9,10))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_SignalGoalJointAction_ReceiverActionInconsistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertFalse(consistencyBoolean)

	@data(('green', {NC.INTENTIONS: 'green triangle', NC.ACTIONS:((7,0),(1,10))}, '1'), 
		('triangle',{NC.INTENTIONS: 'green triangle', NC.ACTIONS:((3,0),(1,10))}, '1'), 
		('green', {NC.INTENTIONS: 'green circle', NC.ACTIONS:((7,0),(1,10))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_SignalGoalJointAction_SignalerActionInconsistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertFalse(consistencyBoolean)

	@data(('green', {NC.INTENTIONS: 'purple circle', NC.ACTIONS:((3,0),(1,10))}, '1'), 
		('triangle',{NC.INTENTIONS: 'green circle', NC.ACTIONS:((7,0),(1,10))}, '1'), 
		('triangle', {NC.INTENTIONS: 'purple circle', NC.ACTIONS:((7,0),(1,10))}, '1'))
	@unpack
	def test_signalIsConsistent_Experiment_SignalGoalJointAction_GoalInconsistent(self, signal, mind, signalerType):
		getSignalConsistency = targetCode.SignalIsConsistent_Experiment(
			signalDictionary = self.signalDictionary_unambiguousSignal, 
			targetDictionary= self.targetDictionary, 
			signalerLocation = self.signalerLocation, 
			receiverLocation=self.receiverLocation)
		consistencyBoolean = getSignalConsistency(signal, mind, signalerType)
		self.assertFalse(consistencyBoolean)



if __name__ == '__main__':
	unittest.main(verbosity=2)
