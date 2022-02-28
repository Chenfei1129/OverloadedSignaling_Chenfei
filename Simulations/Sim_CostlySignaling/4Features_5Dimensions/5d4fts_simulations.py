import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_5_dimensions_4_features_central_controller_round_1.pkl')
data2 = pd.read_pickle('./simulation_5_dimensions_4_features_central_controller_round_2.pkl')
data3 = pd.read_pickle('./simulation_5_dimensions_4_features_central_controller_round_3.pkl')
data4 = pd.read_pickle('./simulation_5_dimensions_4_features_central_controller_round_4.pkl')
data5 = pd.read_pickle('./simulation_5_dimensions_4_features_central_controller_round_5.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
sq_comm3 = data3.loc[data3['CentralControl_actor'] == 'receiver']
sq_comm4 = data4.loc[data4['CentralControl_actor'] == 'receiver']
sq_comm5 = data5.loc[data5['CentralControl_actor'] == 'receiver']
print(sq_comm.shape)
print(sq_comm2.shape)
print(sq_comm3.shape)
print(sq_comm4.shape)
print(sq_comm5.shape)

data = data.append(data2)
data = data.append(data3)
data = data.append(data4)
data = data.append(data5)
data = data.loc[data['CentralControl_actor'] == 'receiver']
data = data[0:1000]
#data.to_pickle('dim5ft4_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
data = pd.read_pickle('dim5ft4_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
print(data.shape)
