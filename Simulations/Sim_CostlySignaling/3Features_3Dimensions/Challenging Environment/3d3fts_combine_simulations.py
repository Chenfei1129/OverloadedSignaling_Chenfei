import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
#from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('simulation_signalerCanQuit_a4_r8_seed282_variedItemFullVocab_Chenfei_three_features_Comparison_More_Added_alpha10_maxSampling.pkl')
data2 = pd.read_pickle('simulation_signalerCanQuit_a4_r8_seed282_variedItemFullVocab_Chenfei_three_features_Comparison_More_Added_alpha10_maxSampling_add.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
print(sq_comm['nTargets'])
print(sq_comm['targetDictionary'])
print(sq_comm['targetDictionary'][14])
print(sq_comm['intention'][14])
#print(sq_comm.shape)
#print(sq_comm2['nTargets'])
#print(sq_comm2['targetDictionary'])
#print(sq_comm2.shape)

data['baseline'] = data.apply(lambda row: findNumberofWordsinBaseline(row) , axis = 1)
print(data['baseline'].mean())
data = data.append(data2)
print(data.shape)
#data.to_pickle('dim4ft5_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs.pkl')
