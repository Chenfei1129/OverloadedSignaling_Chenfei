import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_3_dimensions_5_features_central_controller_round_1.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
data2 = pd.read_pickle('./simulation_3_dimensions_5_features_central_controller_round_2.pkl')
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']
data3 = pd.read_pickle('./simulation_3_dimensions_5_features_central_controller_round_3.pkl')
sq_comm3 = data3.loc[data3['CentralControl_actor'] == 'receiver']
print(sq_comm.shape)
print(sq_comm2.shape)
print(sq_comm3.shape)
data = data.append(data2)
data = data.append(data3)
data = data.loc[data['CentralControl_actor'] == 'receiver']
data = data[0:1000]
#data.to_pickle('./dim5ft5_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
data = pd.read_pickle('./dim5ft5_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
print(data.shape)
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
