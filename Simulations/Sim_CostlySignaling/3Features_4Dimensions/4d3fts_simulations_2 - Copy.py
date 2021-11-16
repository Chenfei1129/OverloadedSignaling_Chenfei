import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_4_dimensions_round__1_central_controller.pkl')
data2 = pd.read_pickle('./simulation_4_dimensions_round__2_central_controller.pkl')
data4 = pd.read_pickle('./simulation_4_dimensions_round__3_central_controller.pkl')
data3 = pd.read_pickle('./dim4ft3_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_06_2000runs.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
sq_comm3 = data3.loc[data3['CentralControl_actor'] == 'receiver']
sq_comm4 = data4.loc[data4['CentralControl_actor'] == 'receiver']
print(sq_comm.shape)
print(sq_comm2.shape)
print(sq_comm3.shape)
print(sq_comm4.shape)


data_result = sq_comm.append(sq_comm2)
data_result = data_result.append(sq_comm3)
data_result = data_result.append(sq_comm4)
print(data_result.shape)
data_result = data_result.loc[data_result['nTargets'] != 4]
data_result = data_result.loc[data_result['nTargets'] != 14]
data_result = data_result.loc[data_result['nTargets'] != 16]
data_result = data_result[0:1000]
print(data_result.shape)
#data_result.to_pickle('./dim4ft3_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_06_2000runs_central_controller.pkl')
