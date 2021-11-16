import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('simulation_signalerCanQuit_a4_r8_seed282_variedItemFullVocab_Chenfei_three_features_Comparison_More_Added_alpha10_maxSampling.pkl')
data2 = pd.read_pickle('simulation_signalerCanQuit_a4_r8_seed282_variedItemFullVocab_Chenfei_three_features_Comparison_More_Added_alpha10_maxSampling_add.pkl')
data_more = pd.read_pickle('./simulation_3_dimensions_3_features_central_controller_round_3.pkl')
data = data.append(data_more)
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
sq_comm = sq_comm.loc[sq_comm['nTargets'] != 4]
sq_comm = sq_comm.loc[sq_comm['nTargets'] != 14]
sq_comm = sq_comm.loc[sq_comm['nTargets'] != 16]
#print(sq_comm2['nTargets'])
#print(sq_comm2['targetDictionary'])
#print(sq_comm2.shape)
print(sq_comm.shape)
data = sq_comm[0:1000]
#data['baseline'] = data.apply(lambda row: findNumberofWordsinBaseline(row) , axis = 1)
#print(data['targetDictionary'])
#print(data.shape)
#print(data2.shape)
#data = data.append(data2)
print(data.shape)
data.to_pickle('dim3ft3_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
