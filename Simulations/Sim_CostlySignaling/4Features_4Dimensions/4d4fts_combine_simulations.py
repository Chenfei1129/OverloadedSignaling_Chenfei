import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_1.pkl')
data2 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_2.pkl')
data4 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_3.pkl')
data3 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_4.pkl')
data5 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_5.pkl')
data6 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_6.pkl')
data7 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_7.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
sq_comm3 = data3.loc[data3['CentralControl_actor'] == 'receiver']
sq_comm4 = data4.loc[data4['CentralControl_actor'] == 'receiver']
sq_comm5 = data5.loc[data5['CentralControl_actor'] == 'receiver']
sq_comm6 = data6.loc[data6['CentralControl_actor'] == 'receiver']
sq_comm7 = data7.loc[data7['CentralControl_actor'] == 'receiver']
print(sq_comm.shape)
print(sq_comm2.shape)
print(sq_comm3.shape)
print(sq_comm4.shape)
print(sq_comm5.shape)
print(sq_comm6.shape)
print(sq_comm7.shape)
#print(sq_comm4)
#print(sq_comm5)

print(sq_comm6['targetDictionary'])
print(sq_comm7['targetDictionary'])
data = data.append(data2)
data = data.append(data3)
data = data.append(data4)
data = data.loc[data['CentralControl_actor'] == 'receiver']
data = data[0:1000]
data.to_pickle('dim4ft4_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
print(data['targetDictionary'][0])
#data['baseline'] = data.apply(lambda row: findNumberofWordsinBaseline(row) , axis = 1)
data = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_round_1.pkl')
sq_comm = data.loc[data['nTargets'] == 6]
print(sq_comm['targetDictionary'][5])
print(sq_comm['intention'][5])
