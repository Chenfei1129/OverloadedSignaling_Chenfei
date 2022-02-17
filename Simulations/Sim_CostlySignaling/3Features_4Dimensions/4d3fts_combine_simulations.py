import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure
from processCostlySignalingSimulation import isDIYBetterThanCCWithCosts, calculateBits2, getUtilityWithSignalCost_base
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


data = sq_comm.append(sq_comm2)
data = data.append(sq_comm3)
data = data.append(sq_comm4)
data = data[:2000]

#data = pd.read_pickle('./dim4ft3_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_06_2000runs.pkl')
data['baseline'] = data.apply(lambda row:  4, axis = 1)
data['SigLength'] = data.apply(lambda row: getSignalLength(row, ''), axis = 1)
data['SigLength_1'] = data.apply(lambda row: getSignalLength(row, 'S_C0.1'), axis = 1)
data['SigLength_2'] = data.apply(lambda row: getSignalLength(row, 'S_C0.2'), axis = 1)
data['SigLength_3'] = data.apply(lambda row: getSignalLength(row, 'S_C0.3'), axis = 1)
data['SigLength_4'] = data.apply(lambda row: getSignalLength(row, 'S_C0.4'), axis = 1)
data['SigLength_5'] = data.apply(lambda row: getSignalLength(row, 'S_C0.5'), axis = 1)
data['SigLength_6'] = data.apply(lambda row: getSignalLength(row, 'S_C0.6'), axis = 1)

data['bits_0'] = data.apply(lambda row: calculateBits2(row, 'SigLength', 4, 3), axis = 1)
data['bits_1'] = data.apply(lambda row: calculateBits2(row, 'SigLength_1', 4, 3), axis = 1)
data['bits_2'] = data.apply(lambda row: calculateBits2(row, 'SigLength_2', 4, 3), axis = 1)
data['bits_3'] = data.apply(lambda row: calculateBits2(row, 'SigLength_3', 4, 3), axis = 1)
data['bits_4'] = data.apply(lambda row: calculateBits2(row, 'SigLength_4', 4, 3), axis = 1)
data['bits_5'] = data.apply(lambda row: calculateBits2(row, 'SigLength_5', 4, 3), axis = 1)
data['bits_6'] = data.apply(lambda row: calculateBits2(row, 'SigLength_6', 4, 3), axis = 1)

data['utility'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0,
                                                                  communicate_columnName = 'IW_S1R0_sChoice' ,
                                                                  cc_columnName = 'CentralControl_utility',
                                                                  diy_columnName = 'DIYSignaler_utility', reach_target_columnName = 'IW_S1R0_sAchievesGoal'), axis = 1)
data['utility_1'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0.1,
                                                                  communicate_columnName = 'IW_S1R0_sChoiceS_C0.1' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.1',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.1', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.1'), axis = 1)
data['utility_2'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0.2,
                                                                  communicate_columnName = 'IW_S1R0_sChoiceS_C0.2' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.2',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.2', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.2'), axis = 1)
data['utility_3'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0.3,
                                                                  communicate_columnName = 'IW_S1R0_sChoiceS_C0.3' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.3',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.3', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.3'), axis = 1)
data['utility_4'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0.4,
                                                                  communicate_columnName = 'IW_S1R0_sChoiceS_C0.4' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.4',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.4', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.4'), axis = 1)
data['utility_5'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0.5,
                                                                  communicate_columnName = 'IW_S1R0_sChoiceS_C0.5' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.5',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.5', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.5'), axis = 1)
data['utility_6'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0.6,
                                                                  communicate_columnName = 'IW_S1R0_sChoiceS_C0.6' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.6',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.6', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.6'), axis = 1)

def getProportionTargetReached(df):
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0
    getPropFalse =  lambda colName: df[colName].value_counts(normalize=True).loc[False] if (False in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerQuits =  lambda colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerDIY_Optimal =  lambda colName: df[colName].value_counts(normalize=True).loc['do_DIY_Optimal'] if ('do_DIY_Optimal' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerDIY_subOptimal =  lambda colName: df[colName].value_counts(normalize=True).loc['do_DIY_suboptimal'] if ('do_DIY_suboptimal' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerCommunication_Fail =  lambda colName: df[colName].value_counts(normalize=True).loc['communication_failure'] if ('communication_failure' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropCom = lambda colName: df[colName].value_counts(normalize=True).loc['Communicate'] if ('Communicate' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropDIY = lambda colName: df[colName].value_counts(normalize=True).loc['DIY'] if ('DIY' in df[colName].value_counts(normalize=True).index) else 0.0
    dfNew = {
             'count_bits_std': [df['bits_0'].std(),df['bits_1'].std(),df['bits_2'].std(), df['bits_3'].std(),df['bits_4'].std(), df['bits_5'].std(), df['bits_6'].std()],
             'count_bits' : [df['bits_0'].mean(),df['bits_1'].mean(),df['bits_2'].mean(), df['bits_3'].mean(),df['bits_4'].mean(), df['bits_5'].mean(), df['bits_6'].mean()],
             'quit':[getPropSignalerQuits('Type_Failure_0'),getPropSignalerQuits('Type_Failure_1'), getPropSignalerQuits('Type_Failure_2'),
                     getPropSignalerQuits('Type_Failure_3'),getPropSignalerQuits('Type_Failure_4'), getPropSignalerQuits('Type_Failure_5'),getPropSignalerQuits('Type_Failure_6')
                     ],
             'DIY_Optimal':[getPropSignalerDIY_Optimal('Type_Failure_0'),getPropSignalerDIY_Optimal('Type_Failure_1'), getPropSignalerDIY_Optimal('Type_Failure_2'),
                     getPropSignalerDIY_Optimal('Type_Failure_3'),getPropSignalerDIY_Optimal('Type_Failure_4'), getPropSignalerDIY_Optimal('Type_Failure_5'),getPropSignalerDIY_Optimal('Type_Failure_6')
                     ],
             'DIY_subOptimal':[getPropSignalerDIY_subOptimal('Type_Failure_0'),getPropSignalerDIY_subOptimal('Type_Failure_1'), getPropSignalerDIY_subOptimal('Type_Failure_2'),
                     getPropSignalerDIY_subOptimal('Type_Failure_3'),getPropSignalerDIY_subOptimal('Type_Failure_4'), getPropSignalerDIY_subOptimal('Type_Failure_5'),getPropSignalerDIY_subOptimal('Type_Failure_6')
                     ],
             'Communication_failure':[getPropSignalerCommunication_Fail('Type_Failure_0'),getPropSignalerCommunication_Fail('Type_Failure_1'), getPropSignalerCommunication_Fail('Type_Failure_2'),
                     getPropSignalerCommunication_Fail('Type_Failure_3'),getPropSignalerCommunication_Fail('Type_Failure_4'), getPropSignalerCommunication_Fail('Type_Failure_5'),getPropSignalerCommunication_Fail('Type_Failure_6')
                     ],
             'Communication':[getPropFalse('Type_Failure_0'),getPropFalse('Type_Failure_1'), getPropFalse('Type_Failure_2'),
                     getPropFalse('Type_Failure_3'),getPropFalse('Type_Failure_4'), getPropFalse('Type_Failure_5'),getPropFalse('Type_Failure_6')
                     ],
             'Communication_baseline':[getPropCom('baseline_0'),getPropCom('baseline_1'), getPropCom('baseline_2'),
                     getPropCom('baseline_3'),getPropCom('baseline_4'), getPropCom('baseline_5'),getPropCom('baseline_6')
                     ],
             'quit_baseline':[getPropSignalerQuits('baseline_0'),getPropSignalerQuits('baseline_1'), getPropSignalerQuits('baseline_2'),
                     getPropSignalerQuits('baseline_3'),getPropSignalerQuits('baseline_4'), getPropSignalerQuits('baseline_5'),getPropSignalerQuits('baseline_6')
                     ],
             'DIY_baseline':[getPropDIY('baseline_0'),getPropDIY('baseline_1'), getPropDIY('baseline_2'),
                     getPropDIY('baseline_3'),getPropDIY('baseline_4'), getPropDIY('baseline_5'),getPropDIY('baseline_6')
                     ],
             'utility':[df['utility'].mean(),df['utility_1'].mean(),df['utility_2'].mean(), df['utility_3'].mean(),df['utility_4'].mean(),df['utility_5'].mean(),df['utility_6'].mean()],
             'utility_base':[df['utility_base'].mean(),df['utility_1_base'].mean(),df['utility_2_base'].mean(),
                             df['utility_3_base'].mean(),df['utility_4_base'].mean(),df['utility_5_base'].mean(),df['utility_6_base'].mean()],
             'utility_percent':[(df['utility']/df['CentralControl_utility']).mean(),(df['utility_1']/df['CentralControl_utilityS_C0.1']).mean(),
                        (df['utility_2']/df['CentralControl_utilityS_C0.2']).mean(), (df['utility_3']/df['CentralControl_utilityS_C0.3']).mean(),
                        (df['utility_4']/df['CentralControl_utilityS_C0.4']).mean(),
                        (df['utility_5']/df['CentralControl_utilityS_C0.5']).mean(),(df['utility_6']/df['CentralControl_utilityS_C0.6']).mean()],
             'utility_base_percent':[(df['utility_base']/df['CentralControl_utility']).mean(),
                             (df['utility_1_base']/df['CentralControl_utilityS_C0.1']).mean(),(df['utility_2_base']/df['CentralControl_utilityS_C0.2']).mean(),
                            (df['utility_3_base']/df['CentralControl_utilityS_C0.3']).mean(),
                             (df['utility_4_base']/df['CentralControl_utilityS_C0.4']).mean(),
                             (df['utility_5_base']/df['CentralControl_utilityS_C0.5']).mean(),(df['utility_6_base']/df['CentralControl_utilityS_C0.6']).mean()],
             'utility_percent_std':[(df['utility']/df['CentralControl_utility']).std(),(df['utility_1']/df['CentralControl_utilityS_C0.1']).std(),
                        (df['utility_2']/df['CentralControl_utilityS_C0.2']).std(), (df['utility_3']/df['CentralControl_utilityS_C0.3']).std(),
                        (df['utility_4']/df['CentralControl_utilityS_C0.4']).std(),
                        (df['utility_5']/df['CentralControl_utilityS_C0.5']).std(),(df['utility_6']/df['CentralControl_utilityS_C0.6']).std()],
             'utility_base_percent_std':[(df['utility_base']/df['CentralControl_utility']).std(),
                             (df['utility_1_base']/df['CentralControl_utilityS_C0.1']).std(),(df['utility_2_base']/df['CentralControl_utilityS_C0.2']).std(),
                            (df['utility_3_base']/df['CentralControl_utilityS_C0.3']).std(),
                             (df['utility_4_base']/df['CentralControl_utilityS_C0.4']).std(),
                             (df['utility_5_base']/df['CentralControl_utilityS_C0.5']).std(),(df['utility_6_base']/df['CentralControl_utilityS_C0.6']).std()],
             }
    propTrialsTargetReached = pd.DataFrame(dfNew, columns = ['count_bits', 'count_bits_std', 'quit', 'DIY_Optimal', 'DIY_subOptimal', 'Communication_failure', 'Communication', 'utility', 'Communication_baseline', 'quit_baseline',
                                                             'DIY_baseline', 'utility_base', 'utility_percent', 'utility_base_percent', 'utility_percent_std', 'utility_base_percent_std' 
                                                             ], index=['IW_S1R0', 'IW_S1R0S_C0.1','IW_S1R0S_C0.2', 'IW_S1R0S_C0.3','IW_S1R0S_C0.4', 'IW_S1R0S_C0.5','IW_S1R0S_C0.6' ])

    #propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorStd'] =  1.96*np.sqrt(propTrialsTargetReached['count_bits_std'] * propTrialsTargetReached['count_bits_std'] /df.shape[0])
    propTrialsTargetReached['marginOfErrorStd_utility_percent_base'] =  1.96*np.sqrt(propTrialsTargetReached['utility_base_percent_std' ] * propTrialsTargetReached['utility_base_percent_std' ] /df.shape[0])
    propTrialsTargetReached['marginOfErrorStd_utility_percent'] =  1.96*np.sqrt(propTrialsTargetReached['utility_percent_std' ] * propTrialsTargetReached['utility_percent_std'] /df.shape[0])
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

data['utility_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0,
                                                                  communicate_columnName = 'baseline_0' ,
                                                                  cc_columnName = 'CentralControl_utility',
                                                                  diy_columnName = 'DIYSignaler_utility', reach_target_columnName = 'IW_S1R0_sAchievesGoal', numBase = 4), axis = 1)
data['utility_1_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0.1,
                                                                  communicate_columnName = 'baseline_1' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.1',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.1', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.1', numBase = 4), axis = 1)
data['utility_2_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0.2,
                                                                  communicate_columnName = 'baseline_2' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.2',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.2', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.2', numBase = 4), axis = 1)
data['utility_3_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0.3,
                                                                  communicate_columnName = 'baseline_3' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.3',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.3', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.3', numBase = 4), axis = 1)
data['utility_4_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0.4,
                                                                  communicate_columnName = 'baseline_4' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.4',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.4', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.4', numBase = 4), axis = 1)
data['utility_5_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0.5,
                                                                  communicate_columnName = 'baseline_5' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.5',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.5', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.5', numBase = 4), axis = 1)
data['utility_6_base'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, signalCost = 0.6,
                                                                  communicate_columnName = 'baseline_6' ,
                                                                  cc_columnName = 'CentralControl_utilityS_C0.6',
                                                                  diy_columnName = 'DIYSignaler_utilityS_C0.6', reach_target_columnName = 'IW_S1R0_sAchievesGoalS_C0.6', numBase = 4), axis = 1)


sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']

def getTypeDifference(dfrow, baseColumn, column):
    if dfrow[baseColumn] == 'quit' and dfrow[column] == 'quit' :
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
#print((data['utility_1']/data['CentralControl_utilityS_C0.1']).mean())
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']

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
"""
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_6'] >0]

k = ['4', '6', '8', '10', '12', '14', '16']
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_6'] >0]
acc0, accDF0, acc1, accDF1, acc2, accDF2, acc3, accDF3, acc4, accDF4, acc5, accDF5, acc6, accDF6 = facetByItem(sq_comm)

base = 16
base_line = [base, base, base, base, base, base, base]

plt.errorbar(k, base_line, color = 'black', label = 'baseline')
plt.errorbar(acc0.keys(), acc0.values(), yerr = accDF0.values(), color = 'green', label = 'IW Cost = 0')
plt.errorbar(acc1.keys(), acc1.values(), yerr = accDF1.values(), color = 'red', label = 'IW cost = 0.1')
plt.errorbar(acc2.keys(), acc2.values(), yerr = accDF2.values(), color = 'yellow', label = 'IW cost = 0.2')
plt.errorbar(acc3.keys(), acc3.values(), yerr = accDF3.values(), color = 'orange', label = 'IW cost = 0.3')
plt.errorbar(acc4.keys(), acc4.values(), yerr = accDF4.values(), color = 'grey', label = 'IW cost = 0.4')
plt.errorbar(acc5.keys(), acc5.values(), yerr = accDF5.values(), color = 'purple', label = 'IW cost = 0.5')
plt.errorbar(acc6.keys(), acc6.values(), yerr = accDF6.values(), color = 'blue', label = 'IW cost = 0.6')
sns.despine()

plt.ylim(0, 16)
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
plt.legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Number of Items', fontsize = 25)
plt.ylabel('Average Bits', fontsize = 25)
sns.despine()
#plt.xlabel('Signal Cots Per Word')
#Set Quit or DIY Equlas to 0
plt.title ('Average Bits --- Quit or DIY = 0', fontsize = 25)#Remove All Rows that Any Cost Choose to Quit or DIY

plt.show()
"""
"""
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_6'] >0]

k = ['4', '6', '8', '10', '12', '14', '16']
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_6'] >0]
bits = getProportionTargetReached(sq_commCount)['count_bits']
bits_std = getProportionTargetReached(sq_commCount)['marginOfErrorStd']
Percent_Bits_Conserved = (16-bits[0:2])*100/16
#If remove all rows the contain Quit or DIY.
 # 1071 rows
#print(sq_commCount.shape)
print(sq_comm.shape)
g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6']
m = [16, 16, 16, 16, 16, 16, 16]
plt.errorbar(g, bits, yerr = bits_std, label = 'Imagined We Model', color = '#ad0b2d', linewidth = 4 )
plt.errorbar(g, m, label = 'Baseline Model',  color = '#555555', linewidth = 4 )
plt.ylim(0, 16)
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
plt.legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Signal Cost Per Word', fontsize = 25)
plt.ylabel('Average Bits', fontsize = 25)
sns.despine()
#plt.xlabel('Signal Cots Per Word')
plt.ylabel('Average Bits')
#Set Quit or DIY Equlas to 0
plt.title ('Average Bits vs Signal Cost Per Word', fontsize = 25)#'Average Bits --- Remove All Rows that Any Cost Choose to Quit or DIY'
#plt.show()
"""
"""
k = ['4', '6', '8', '10', '12', '14', '16']
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_6'] >0]
"""
"""
quit_total = getProportionTargetReached(sq_comm)['quit'] * 100
DIY_small = getProportionTargetReached(sq_comm)['DIY_Optimal'] * 100
DIY_large = getProportionTargetReached(sq_comm)['DIY_subOptimal'] * 100
receiver_failed = getProportionTargetReached(sq_comm)['Communication_failure'] * 100
print(getProportionTargetReached(sq_comm)['Communication'])
print(DIY_small)
print(quit_total)
import seaborn as sns 
g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
fig = plt.figure(figsize = (16, 5))
#ax[0].legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.bar(g, quit_total, label = 'Signaler Quits', edgecolor = 'white', color = '#000650')
plt.bar(g, DIY_small, bottom = quit_total, label ='Signaler Does for Self(perferred given signal cost)',edgecolor='white', color = '#91AFEC' )
plt.bar(g, DIY_large,  label = 'Signaler Does for Self(suboptimal)', edgecolor='white'  )
plt.bar(g, receiver_failed,  label = 'Communication Fails')
plt.xlabel('Signal Cost Per Word', size = 26)
plt.ylabel('Percent of Goal \nnot Reached by Receiver', size = 26)
plt.legend()
plt.ylim(0, 100)
plt.title ('IW Model', size = 26)
sns.despine()
#plt.set(ylabel = 'Percent of Utility Achieved', ylim = (0, 100))
#plt.legend()
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
#plt.show()
#plt.xlabel('Signal Cots Per Word')
filename = 'Signal Cost vs Percent of Goal Not Reached IW'
fig.savefig(filename, dpi = 300, bbox_inches = 'tight')
#plt.ylabel('Percent of Utility Achieved', ylim = (0, 100))

plt.show()
"""
"""
quit_total = getProportionTargetReached(sq_comm)['quit_baseline'] * 100
DIY = getProportionTargetReached(sq_comm)['DIY_baseline'] * 100
receiver_failed = getProportionTargetReached(sq_comm)['Communication_failure'] * 100
print(getProportionTargetReached(sq_comm)['Communication'])
print(quit_total)
print(DIY)
import seaborn as sns

g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
fig = plt.figure(figsize = (16, 5))
#ax[0].legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#plt.bar(g, DIY_large,label = 'Signaler Does for Self(suboptimal)', edgecolor='white'  )
plt.bar(g, quit_total, label = 'Signaler Quits', edgecolor = 'white', color = '#000650')
plt.bar(g, DIY, label ='Signaler Does for Self(perferred given signal cost)',edgecolor='white', bottom = quit_total, color = '#91AFEC' )

#plt.bar(g, receiver_failed, label = 'Communication Fails')
plt.xlabel('Signal Cost Per Word', size = 26)
plt.ylabel('Percent of Goal \nnot Reached by Receiver', size = 26)
plt.legend()
plt.title ('Baseline Model', size = 26)
plt.ylim(0, 100)
sns.despine()

#plt.set(ylabel = 'Percent of Utility Achieved', ylim = (0, 100))
#plt.legend()
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
#plt.show()
#plt.xlabel('Signal Cots Per Word')
filename = 'Signal Cost vs Percent of Goal Not Reached Baseline'
fig.savefig(filename, dpi = 300, bbox_inches = 'tight')
#plt.ylabel('Percent of Utility Achieved', ylim = (0, 100))

plt.show()
"""
#print(-len(list((sq_comm[communicate_columnName].split()))))
#print(sq_comm['CentralControl_utilityS_C0.1'][:3])
#print(sq_comm['IW_S1R0_sChoiceS_C0.1'][:3])
#print(sq_comm['utility_1_base'][:3])
#print(sq_comm['utility_1'][:3])
quit_total = getProportionTargetReached(sq_comm)['quit_baseline'] * 100
quit_IW = getProportionTargetReached(sq_comm)['quit'] * 100
DIY = getProportionTargetReached(sq_comm)['DIY_baseline'] * 100
DIY_large = getProportionTargetReached(sq_comm)['DIY_Optimal']*100
receiver_failed = getProportionTargetReached(sq_comm)['Communication_failure'] * 100
print(getProportionTargetReached(sq_comm)['Communication'])
print(quit_total)
print(DIY)
import seaborn as sns

#g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
g = np.arange(7) 
width = 0.35
fig = plt.figure(figsize = (16, 5))
fig, ax = plt.subplots()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#plt.bar(g, DIY_large,label = 'Signaler Does for Self(suboptimal)', edgecolor='white'  )
ax.bar(g-width/2, quit_total, width, label = 'Signaler Quits --- Baseline', edgecolor = 'white', color = '#000650', alpha = 0.5)
ax.bar(g-width/2, DIY, width, label ='Signaler Does for Self(perferred given signal cost) --- Baseline',edgecolor='white', bottom = quit_total, color = '#91AFEC', alpha = 0.5 )
ax.bar(g+width/2, quit_IW, width, label = 'Signaler Quits --- IW', edgecolor = 'white', color = '#000650')
ax.bar(g+width/2, DIY_large, width, label ='Signaler Does for Self(perferred given signal cost) --- IW',edgecolor='white', bottom = quit_IW, color = '#91AFEC')
#plt.bar(g, receiver_failed, label = 'Communication Fails')
ax.set_xlabel('Signal Cost Per Word', size = 26)
ax.set_ylabel('Percent of Goal \nnot Reached by Receiver', size = 26)
ax.legend()
ax.set_title ('Percent of Goal not Reached by Receiver vs Signal Cost Per Word', size = 26)
ax.set_ylim(0, 100)
sns.despine()
ax.set_xticks(g)
ax.set_xticklabels(('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'))
#plt.set_xticklabels(('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'))
#plt.set(ylabel = 'Percent of Utility Achieved', ylim = (0, 100))
#plt.legend()
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
#plt.show()
#plt.xlabel('Signal Cots Per Word')
filename = 'Signal Cost vs Percent of Goal Not Reached Baseline and IW'
figure = plt.gcf()
figure.set_size_inches(16, 5.5)
#plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
#plt.ylabel('Percent of Utility Achieved', ylim = (0, 100))
def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')




plt.show()
"""
print(getProportionTargetReached(sq_comm)['utility_base_percent'])
print(getProportionTargetReached(sq_comm)['utility_percent'])
print(getProportionTargetReached(sq_comm)['marginOfErrorStd_utility_percent_base'])
print(getProportionTargetReached(sq_comm)['marginOfErrorStd_utility_percent'])

Utility_percent = getProportionTargetReached(sq_comm)['utility_percent'] * 100
Utility_percent_base = getProportionTargetReached(sq_comm)['utility_base_percent'] * 100
utility_error = getProportionTargetReached(sq_comm)['marginOfErrorStd_utility_percent'] * 100
utility_error_base = getProportionTargetReached(sq_comm)['marginOfErrorStd_utility_percent_base'] * 100
#print(Utility_percent)
g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6']
#m = [16, 16, 16, 16, 16, 16, 16]
plt.errorbar(g,Utility_percent , yerr = utility_error, label = 'Imagined We Model', color = '#ad0b2d', linewidth = 4 )
plt.errorbar(g,Utility_percent_base , yerr = utility_error_base, label = 'Baseline Model',  color = '#555555', linewidth = 4 )
plt.ylim(0, 110)
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
plt.legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Signal Cost Per Word', fontsize = 25)
plt.ylabel('Percent of Utility Achieved', fontsize = 25)
sns.despine()
#plt.xlabel('Signal Cots Per Word')
#plt.ylabel('Percent of Bits Conserved')
#Set Quit or DIY Equlas to 0
plt.title ('Percent of Utility Achieved vs Signal Cost Per Word', fontsize = 25)# --- Remove Rows that the Cost Choose to Quit or DIY
plt.show()
"""
"""
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount1 = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount2 = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount3 = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount4 = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount5 = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount6 = sq_commCount.loc[sq_commCount['bits_6'] >0]


bits0 = getProportionTargetReached(sq_commCount)['count_bits'][0]
bits1 = getProportionTargetReached(sq_commCount1)['count_bits'][1]
bits2 = getProportionTargetReached(sq_commCount2)['count_bits'][2]
bits3 = getProportionTargetReached(sq_commCount3)['count_bits'][3]
bits4 = getProportionTargetReached(sq_commCount4)['count_bits'][4]
bits5 = getProportionTargetReached(sq_commCount5)['count_bits'][5]
bits6 = getProportionTargetReached(sq_commCount6)['count_bits'][6]

bits00 = getProportionTargetReached(sq_comm)['count_bits'][0]
bits01 = getProportionTargetReached(sq_comm)['count_bits'][1]
bits_std0 = getProportionTargetReached(sq_comm)['marginOfErrorStd'][0]
bits_std1 = getProportionTargetReached(sq_comm)['marginOfErrorStd'][1]
#If remove all rows the contain Quit or DIY.
 # 1071 rows
#print(sq_commCount.shape)
bits = [(16-bits00)/16*100, (16-bits01)/16*100]#, (16-bits2)/16*100, (16-bits3)/16*100, (16-bits4)/16*100, (16-bits5)/16*100, (16-bits6)/16*100
bits_error = [bits_std0*100/16, bits_std1*100/16]
print(bits_error)
cost = ['0', '0.1']
g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6']
m = [16, 16, 16, 16, 16, 16, 16]
plt.bar(cost, bits, yerr = bits_error, label = 'Imagined We Model', color = '#ad0b2d', linewidth = 4 )
#plt.errorbar(g, m, label = 'Baseline Model',  color = '#555555', linewidth = 4 )
plt.ylim(0, 100)
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
plt.legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Signal Cost Per Word', fontsize = 25)
plt.ylabel('Percent of Bits Conserved', fontsize = 25)
sns.despine()
#plt.xlabel('Signal Cots Per Word')
plt.ylabel('Percent of Bits Conserved')
#Set Quit or DIY Equlas to 0
plt.title ('Percent of Bits Conserved vs Signal Cost Per Word', fontsize = 25)# --- Remove Rows that the Cost Choose to Quit or DIY
plt.show()
"""
