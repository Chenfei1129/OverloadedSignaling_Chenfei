import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_5_dimensions_3_features_central_controller_round_2.pkl')
data2 = pd.read_pickle('./simulation_5_dimensions_3_features_central_controller_round_3.pkl')
data3 = pd.read_pickle('./simulation_5_dimensions_3_features_central_controller_round_4.pkl')
data4 = pd.read_pickle('./simulation_5_dimensions_3_features_central_controller_round_5.pkl')
data5 = pd.read_pickle('./simulation_5_dimensions_3_features_central_controller_round_6.pkl')
data6 = pd.read_pickle('./simulation_5_dimensions_3_features_central_controller_round_7.pkl')
#data['baseline'] = data.apply(lambda row:  5, axis = 1)
#data['SigLength'] = data.apply(lambda row: getSignalLength(row, ''), axis = 1)
#data['SigLength_1'] = data.apply(lambda row: getSignalLength(row, 'S_C0.1'), axis = 1)
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
sq_comm3 = data3.loc[data3['CentralControl_actor'] == 'receiver']
sq_comm4 = data4.loc[data4['CentralControl_actor'] == 'receiver']
sq_comm5 = data5.loc[data5['CentralControl_actor'] == 'receiver']
sq_comm6 = data6.loc[data6['CentralControl_actor'] == 'receiver']
print(sq_comm.shape)
print(sq_comm2.shape)
print(sq_comm3.shape)
print(sq_comm4.shape)
print(sq_comm5.shape)
print(sq_comm6.shape)
print(data2.shape)
#print(sq_comm['SigLength'].mean())
#print(sq_comm['SigLength_1'].mean())
print(data)
print(data5)
data = data.append(data2)
data = data.append(data3)
data = data.append(data4)
data = data.append(data5)
data = data.append(data6)
data = data.loc[data['CentralControl_actor'] == 'receiver'][0:1000]
print(data.shape)
data.to_pickle('dim5ft3_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
