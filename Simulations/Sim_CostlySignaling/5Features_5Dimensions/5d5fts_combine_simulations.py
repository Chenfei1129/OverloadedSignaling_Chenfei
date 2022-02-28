import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_5_dimensions_5_features_central_controller_round_1.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
data2 = pd.read_pickle('./simulation_5_dimensions_5_features_central_controller_round_2.pkl')
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
data3 = pd.read_pickle('./simulation_5_dimensions_5_features_central_controller_round_4.pkl')
sq_comm3 = data3.loc[data3['CentralControl_actor'] == 'receiver']
data4 = pd.read_pickle('./simulation_5_dimensions_5_features_central_controller_round_test.pkl')
sq_comm4 = data4.loc[data4['CentralControl_actor'] == 'receiver']
data5 = pd.read_pickle('./simulation_5_dimensions_5_features_central_controller_round_5.pkl')
sq_comm5 = data5.loc[data5['CentralControl_actor'] == 'receiver']
data6 = pd.read_pickle('./simulation_5_dimensions_5_features_central_controller_round_6.pkl')
sq_comm6 = data6.loc[data6['CentralControl_actor'] == 'receiver']
data = data.append(data2)
data = data.append(data3)
data = data.append(data4)
data = data.append(data5)
data = data.append(data6)
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver'][0:1000]
sq_comm.to_pickle('dim5ft5_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_1000runs_central_controller_receiver.pkl')
print(sq_comm.shape)
"""
print(sq_comm.shape)
print(sq_comm2.shape)
print(sq_comm['targetDictionary'][4])
print(sq_comm2['targetDictionary'][5])
print(sq_comm['intention'][4])
print(sq_comm2['intention'][5])
print(sq_comm['intention'][4])
print(sq_comm2['intention'][5])
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt

sq_comm['baseline'] = data.apply(lambda row:  5, axis = 1)
sq_comm['SigLength'] = data.apply(lambda row: getSignalLength(row, ''), axis = 1)
sq_comm['SigLength_1'] = data.apply(lambda row: getSignalLength(row, 'S_C0.1'), axis = 1)
print(sq_comm['SigLength'].mean())
print(sq_comm['SigLength_1'].mean())
"""
