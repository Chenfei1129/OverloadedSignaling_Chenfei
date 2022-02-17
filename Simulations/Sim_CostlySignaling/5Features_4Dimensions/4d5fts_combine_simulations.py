import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_4_dimensions_5_features_central_controller_round_1.pkl')
data2 = pd.read_pickle('./simulation_4_dimensions_5_features_central_controller_round_2.pkl')
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm2 = data2.loc[data2['CentralControl_actor'] == 'receiver']

print(sq_comm.shape)
#print(sq_comm2['nTargets'])
#print(sq_comm2['targetDictionary'])
print(sq_comm2.shape)

data = sq_comm.append(sq_comm2)
data = data[0:1000]
#data.to_pickle('./.dim4ft5_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller_receiver.pkl')
data = pd.read_pickle('./dim4ft5_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller_receiver.pkl')
print(data.shape)
