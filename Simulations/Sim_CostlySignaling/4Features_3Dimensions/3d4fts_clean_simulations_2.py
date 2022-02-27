import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, baselineChoice, getUtilityWithSignalCost_base
import matplotlib.pyplot as plt
data = pd.read_pickle('./dim3ft4_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_06_2000runs.pkl')
print(data.shape)
data['baseline'] = data.apply(lambda row:  3, axis = 1)
data['SigLength'] = data.apply(lambda row: getSignalLength(row, ''), axis = 1)
data['SigLength_1'] = data.apply(lambda row: getSignalLength(row, 'S_C0.1'), axis = 1)
data['SigLength_2'] = data.apply(lambda row: getSignalLength(row, 'S_C0.2'), axis = 1)
data['SigLength_3'] = data.apply(lambda row: getSignalLength(row, 'S_C0.3'), axis = 1)
data['SigLength_4'] = data.apply(lambda row: getSignalLength(row, 'S_C0.4'), axis = 1)
data['SigLength_5'] = data.apply(lambda row: getSignalLength(row, 'S_C0.5'), axis = 1)
data['SigLength_6'] = data.apply(lambda row: getSignalLength(row, 'S_C0.6'), axis = 1)

data['bits_0'] = data.apply(lambda row: calculateBits(row, 'SigLength', 4, 3), axis = 1)
data['bits_1'] = data.apply(lambda row: calculateBits(row, 'SigLength_1', 4, 3), axis = 1)
data['bits_2'] = data.apply(lambda row: calculateBits(row, 'SigLength_2', 4, 3), axis = 1)
data['bits_3'] = data.apply(lambda row: calculateBits(row, 'SigLength_3', 4, 3), axis = 1)
data['bits_4'] = data.apply(lambda row: calculateBits(row, 'SigLength_4', 4, 3), axis = 1)
data['bits_5'] = data.apply(lambda row: calculateBits(row, 'SigLength_5', 4, 3), axis = 1)
data['bits_6'] = data.apply(lambda row: calculateBits(row, 'SigLength_6', 4, 3), axis = 1)

data['utility'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0,
                                                                  communicate_columnName = 'IW_S1R0_sChoice' ,
                                                                  cc_columnName = 'CentralControl_utility',
                                                                  diy_columnName = 'DIYSignaler_utility', reach_target_columnName = 'IW_S1R0_sAchievesGoal'), axis = 1)

def getProportionTargetReached(df):
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerQuits =  lambda colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0  
    dfNew = {
             'count_bits_std': [df['bits_0'].std(),df['bits_1'].std(),df['bits_2'].std(), df['bits_3'].std(),df['bits_4'].std(), df['bits_5'].std(), df['bits_6'].std()],
             'count_bits' : [df['bits_0'].mean(),df['bits_1'].mean(),df['bits_2'].mean(), df['bits_3'].mean(),df['bits_4'].mean(), df['bits_5'].mean(), df['bits_6'].mean()]
             
             
             }
    propTrialsTargetReached = pd.DataFrame(dfNew, columns = ['count_bits', 'count_bits_std'], index=['IW_S1R0', 'IW_S1R0S_C0.1','IW_S1R0S_C0.2', 'IW_S1R0S_C0.3','IW_S1R0S_C0.4', 'IW_S1R0S_C0.5','IW_S1R0S_C0.6' ])

    #propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorStd'] =  1.96*np.sqrt(propTrialsTargetReached['count_bits_std'] * propTrialsTargetReached['count_bits_std'] /df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd_base'] =  1.96*np.sqrt(propTrialsTargetReached['count_base_std'] * propTrialsTargetReached['count_base_std'] /df.shape[0])    
    #propTrialsTargetReached['receiver failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler'])
    #propTrialsTargetReached['receiver_failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler_does'])
    return propTrialsTargetReached
def facetByItem(df2):
    acc0 = {}
    accDF0 = {}
    acc1 = {}
    accDF1 = {}
    acc2 = {}
    accDF2 = {}
    acc3 = {}
    accDF3 = {}
    acc4 = {}
    accDF4 = {}
    acc5 = {}
    accDF5 = {}
    acc6 = {}
    accDF6 = {}
    for i in [4,6,8,10,12,14,16]:
        df = df2.loc[df2['nTargets'] == i]
        k = getProportionTargetReached(df)
        acc0[str(i)] = k['count_bits'][0]
        accDF0[str(i)] = k['marginOfErrorStd'][0]
        acc1[str(i)] = k['count_bits'][1]
        accDF1[str(i)] = k['marginOfErrorStd'][1]
        acc2[str(i)] = k['count_bits'][2]
        accDF2[str(i)] = k['marginOfErrorStd'][2]
        acc3[str(i)] = k['count_bits'][3]
        accDF3[str(i)] = k['marginOfErrorStd'][3]
        acc4[str(i)] = k['count_bits'][4]
        accDF4[str(i)] = k['marginOfErrorStd'][4]
        acc5[str(i)] = k['count_bits'][5]
        accDF5[str(i)] = k['marginOfErrorStd'][5]
        acc6[str(i)] = k['count_bits'][6]
        accDF6[str(i)] = k['marginOfErrorStd'][6]
    return acc0, accDF0, acc1, accDF1, acc2, accDF2, acc3, accDF3, acc4, accDF4, acc5, accDF5, acc6, accDF6

data['DIY_better_0'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0, 'baseline'), axis = 1)
data['DIY_better_1'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.1, 'baseline'), axis = 1)
data['DIY_better_2'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.2, 'baseline'), axis = 1)
data['DIY_better_3'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.3, 'baseline'), axis = 1)
data['DIY_better_4'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.4, 'baseline'), axis = 1)
data['DIY_better_5'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.5, 'baseline'), axis = 1)
data['DIY_better_6'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.6, 'baseline'), axis = 1)

data['baseline_0'] = data.apply(lambda row: baselineChoice(row, 0, 'baseline'), axis = 1)
data['baseline_1'] = data.apply(lambda row: baselineChoice(row, 0.1, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.1', diy_columnName='DIYSignaler_utilityS_C0.1'), axis = 1)
data['baseline_2'] = data.apply(lambda row: baselineChoice(row, 0.2, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.2', diy_columnName='DIYSignaler_utilityS_C0.2'), axis = 1)
data['baseline_3'] = data.apply(lambda row: baselineChoice(row, 0.3, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.3', diy_columnName='DIYSignaler_utilityS_C0.3'), axis = 1)
data['baseline_4'] = data.apply(lambda row: baselineChoice(row, 0.4, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.4', diy_columnName='DIYSignaler_utilityS_C0.4'), axis = 1)
data['baseline_5'] = data.apply(lambda row: baselineChoice(row, 0.5, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.5', diy_columnName='DIYSignaler_utilityS_C0.5'), axis = 1)
data['baseline_6'] = data.apply(lambda row: baselineChoice(row, 0.6, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.6', diy_columnName='DIYSignaler_utilityS_C0.6'), axis = 1)

def getTypeDifference(dfrow, baseColumn, column):
    if dfrow[baseColumn] == 'quit' and dfrow[column] == False:
        return True
    return False
data['Type_Failure_0'] = data.apply(lambda row: getTypeOfReceiverFailure(row), axis = 1)
data['Type_Failure_1'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.1', 'DIY_better_1', 'IW_S1R0_goalAchievedS_C0.1'), axis = 1)
data['Type_Failure_2'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.2', 'DIY_better_2', 'IW_S1R0_goalAchievedS_C0.2'), axis = 1)
data['Type_Failure_3'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.3', 'DIY_better_3', 'IW_S1R0_goalAchievedS_C0.3'), axis = 1)
data['Type_Failure_4'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.4', 'DIY_better_4', 'IW_S1R0_goalAchievedS_C0.4'), axis = 1)
data['Type_Failure_5'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.5', 'DIY_better_5', 'IW_S1R0_goalAchievedS_C0.5'), axis = 1)
data['Type_Failure_6'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.6', 'DIY_better_6', 'IW_S1R0_goalAchievedS_C0.6'), axis = 1)

data['Type_Difference_2'] = data.apply(lambda row: getTypeDifference(row, 'baseline_2', 'Type_Failure_2'), axis = 1)
data['Type_Difference_3'] = data.apply(lambda row: getTypeDifference(row, 'baseline_3', 'Type_Failure_3'), axis = 1)
data['Type_Difference_4'] = data.apply(lambda row: getTypeDifference(row, 'baseline_4', 'Type_Failure_4'), axis = 1)
data['Type_Difference_5'] = data.apply(lambda row: getTypeDifference(row, 'baseline_5', 'Type_Failure_5'), axis = 1)
data['Type_Difference_6'] = data.apply(lambda row: getTypeDifference(row, 'baseline_6', 'Type_Failure_6'), axis = 1)

data['utility_0'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0, 'IW_S1R0_sChoice', 'CentralControl_utility', 'DIYSignaler_utility', 'IW_S1R0_goalAchieved'), axis = 1)
data['utility_1'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0.1, 'IW_S1R0_sChoiceS_C0.1', 'CentralControl_utilityS_C0.1', 'DIYSignaler_utilityS_C0.1', 'IW_S1R0_goalAchievedS_C0.1'), axis = 1)                              
data['utility_2'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0.2, 'IW_S1R0_sChoiceS_C0.2', 'CentralControl_utilityS_C0.2', 'DIYSignaler_utilityS_C0.2', 'IW_S1R0_goalAchievedS_C0.2'), axis = 1)
data['utility_3'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0.3, 'IW_S1R0_sChoiceS_C0.3', 'CentralControl_utilityS_C0.3', 'DIYSignaler_utilityS_C0.3', 'IW_S1R0_goalAchievedS_C0.3'), axis = 1)
data['utility_4'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0.4, 'IW_S1R0_sChoiceS_C0.4', 'CentralControl_utilityS_C0.4', 'DIYSignaler_utilityS_C0.4', 'IW_S1R0_goalAchievedS_C0.4'), axis = 1)                               
data['utility_5'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0.5, 'IW_S1R0_sChoiceS_C0.5', 'CentralControl_utilityS_C0.5', 'DIYSignaler_utilityS_C0.5', 'IW_S1R0_goalAchievedS_C0.5'), axis = 1)
data['utility_6'] = data.apply(lambda row: getUtilityWithSignalCost(row, 0.6, 'IW_S1R0_sChoiceS_C0.6', 'CentralControl_utilityS_C0.6', 'DIYSignaler_utilityS_C0.6', 'IW_S1R0_goalAchievedS_C0.6'), axis = 1) 
#data['utility_0_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0, 'baseline_0', 'CentralControl_utility', 'DIYSignaler_utility', 'IW_S1R0_goalAchieved'), axis = 1)
#data['utility_1_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.1, 'baseline_1', 'CentralControl_utilityS_C0.1', 'DIYSignaler_utilityS_C0.1', 'IW_S1R0_goalAchievedS_C0.1'), axis = 1)                              
#data['utility_2_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.2, 'baseline_2', 'CentralControl_utilityS_C0.2', 'DIYSignaler_utilityS_C0.2', 'IW_S1R0_goalAchievedS_C0.2'), axis = 1)
#data['utility_3_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.3, 'baseline_3', 'CentralControl_utilityS_C0.3', 'DIYSignaler_utilityS_C0.3', 'IW_S1R0_goalAchievedS_C0.3'), axis = 1)
data['utility_4_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.4, 'baseline_4', 'CentralControl_utilityS_C0.4', 'DIYSignaler_utilityS_C0.4', 'IW_S1R0_goalAchievedS_C0.4'), axis = 1)                               
data['utility_5_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.5, 'baseline_5', 'CentralControl_utilityS_C0.5', 'DIYSignaler_utilityS_C0.5', 'IW_S1R0_goalAchievedS_C0.5'), axis = 1)
data['utility_6_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.6, 'baseline_6', 'CentralControl_utilityS_C0.6', 'DIYSignaler_utilityS_C0.6', 'IW_S1R0_goalAchievedS_C0.6'), axis = 1)  
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm1 = sq_comm.loc[sq_comm['Type_Difference_6'] == True]
print(sq_comm1.shape)
print(sq_comm1['utility_6'].mean())
print(sq_comm1['utility_6_baseline'].mean())
print(sq_comm1['SigLength_6'].mean())
"""
print(sq_comm['Type_Failure_0'].value_counts())
print(sq_comm['Type_Failure_1'].value_counts())
print(sq_comm['Type_Failure_2'].value_counts())
print(sq_comm['Type_Failure_3'].value_counts())
print(sq_comm['Type_Failure_4'].value_counts())
print(sq_comm['Type_Failure_5'].value_counts())
print(sq_comm['Type_Failure_6'].value_counts())

print(sq_comm['baseline_0'].value_counts())
print(sq_comm['baseline_1'].value_counts())
print(sq_comm['baseline_2'].value_counts())
print(sq_comm['baseline_3'].value_counts())
print(sq_comm['baseline_4'].value_counts())
print(sq_comm['baseline_5'].value_counts())
print(sq_comm['baseline_6'].value_counts())

print(sq_comm['Type_Difference_2'].value_counts())
print(sq_comm['Type_Difference_3'].value_counts())
print(sq_comm['Type_Difference_4'].value_counts())
print(sq_comm['Type_Difference_5'].value_counts())
print(sq_comm['Type_Difference_6'].value_counts())
"""
