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
#print(list(data.columns)) IW_S1R0_goalAchieved

data['baseline'] = data.apply(lambda row:  3, axis = 1)
data['SigLength'] = data.apply(lambda row: getSignalLength(row, ''), axis = 1)
data['SigLength_1'] = data.apply(lambda row: getSignalLength(row, 'S_C0.1'), axis = 1)
data['SigLength_2'] = data.apply(lambda row: getSignalLength(row, 'S_C0.2'), axis = 1)
data['SigLength_3'] = data.apply(lambda row: getSignalLength(row, 'S_C0.3'), axis = 1)
data['SigLength_4'] = data.apply(lambda row: getSignalLength(row, 'S_C0.4'), axis = 1)
data['SigLength_5'] = data.apply(lambda row: getSignalLength(row, 'S_C0.5'), axis = 1)
data['SigLength_6'] = data.apply(lambda row: getSignalLength(row, 'S_C0.6'), axis = 1)

data['bits_0'] = data.apply(lambda row: calculateBits(row, 'SigLength', 3, 4), axis = 1)
data['bits_1'] = data.apply(lambda row: calculateBits(row, 'SigLength_1', 3, 4), axis = 1)
data['bits_2'] = data.apply(lambda row: calculateBits(row, 'SigLength_2', 3, 4), axis = 1)
data['bits_3'] = data.apply(lambda row: calculateBits(row, 'SigLength_3', 3, 4), axis = 1)
data['bits_4'] = data.apply(lambda row: calculateBits(row, 'SigLength_4', 3, 4), axis = 1)
data['bits_5'] = data.apply(lambda row: calculateBits(row, 'SigLength_5', 3, 4), axis = 1)
data['bits_6'] = data.apply(lambda row: calculateBits(row, 'SigLength_6', 3, 4), axis = 1)

data['utility'] = data.apply(lambda row: getUtilityWithSignalCost(row, signalCost = 0,
                                                                  communicate_columnName = 'IW_S1R0_sChoice' ,
                                                                  cc_columnName = 'CentralControl_utility',
                                                                  diy_columnName = 'DIYSignaler_utility', reach_target_columnName = 'IW_S1R0_sAchievesGoal'), axis = 1)

data['DIY_better_0'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0, 'baseline'), axis = 1)
data['DIY_better_1'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.1, 'baseline'), axis = 1)
data['DIY_better_2'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.2, 'baseline'), axis = 1)
data['DIY_better_3'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.3, 'baseline'), axis = 1)
data['DIY_better_4'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.4, 'baseline'), axis = 1)
data['DIY_better_5'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.5, 'baseline'), axis = 1)
data['DIY_better_6'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.6, 'baseline'), axis = 1)

data['baseline_DIY'] = data.apply(lambda row: True, axis = 1)
data['baseline_0'] = data.apply(lambda row: baselineChoice(row, 0, 'baseline'), axis = 1)
data['baseline_1'] = data.apply(lambda row: baselineChoice(row, 0.1, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.1', diy_columnName='baseline_DIY'), axis = 1)
data['baseline_2'] = data.apply(lambda row: baselineChoice(row, 0.2, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.2', diy_columnName='baseline_DIY'), axis = 1)
data['baseline_3'] = data.apply(lambda row: baselineChoice(row, 0.3, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.3', diy_columnName='baseline_DIY'), axis = 1)
data['baseline_4'] = data.apply(lambda row: baselineChoice(row, 0.4, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.4', diy_columnName='baseline_DIY'), axis = 1)
data['baseline_5'] = data.apply(lambda row: baselineChoice(row, 0.5, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.5', diy_columnName='baseline_DIY'), axis = 1)
data['baseline_6'] = data.apply(lambda row: baselineChoice(row, 0.6, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.6', diy_columnName='baseline_DIY'), axis = 1)

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
#data['utility_4_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.4, 'baseline_4', 'CentralControl_utilityS_C0.4', 'DIYSignaler_utilityS_C0.4', 'IW_S1R0_goalAchievedS_C0.4'), axis = 1)                               
data['utility_5_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.5, 'baseline_5', 'CentralControl_utilityS_C0.5', 'DIYSignaler_utilityS_C0.5', 'IW_S1R0_goalAchievedS_C0.5'), axis = 1)
#data['utility_6_baseline'] = data.apply(lambda row: getUtilityWithSignalCost_base(row, 0.6, 'baseline_6', 'CentralControl_utilityS_C0.6', 'DIYSignaler_utilityS_C0.6', 'IW_S1R0_goalAchievedS_C0.6'), axis = 1)  

data['Type_Failure_0'] = data.apply(lambda row: getTypeOfReceiverFailure(row), axis = 1)
data['Type_Failure_1'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.1', 'DIY_better_1', 'IW_S1R0_goalAchievedS_C0.1'), axis = 1)
data['Type_Failure_2'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.2', 'DIY_better_2', 'IW_S1R0_goalAchievedS_C0.2'), axis = 1)
data['Type_Failure_3'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.3', 'DIY_better_3', 'IW_S1R0_goalAchievedS_C0.3'), axis = 1)
data['Type_Failure_4'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.4', 'DIY_better_4', 'IW_S1R0_goalAchievedS_C0.4'), axis = 1)
data['Type_Failure_5'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.5', 'DIY_better_5', 'IW_S1R0_goalAchievedS_C0.5'), axis = 1)
data['Type_Failure_6'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.6', 'DIY_better_6', 'IW_S1R0_goalAchievedS_C0.6'), axis = 1)

def getTypeDifference(dfrow, baseColumn, column):
    if dfrow[baseColumn] == 'Communicate' and dfrow[column] == False:
        return True
    return False
data['Type_Difference_4'] = data.apply(lambda row: getTypeDifference(row, 'baseline_4', 'Type_Failure_4'), axis = 1)
data['Type_Difference_5'] = data.apply(lambda row: getTypeDifference(row, 'baseline_5', 'Type_Failure_5'), axis = 1)
data['Type_Difference_6'] = data.apply(lambda row: getTypeDifference(row, 'baseline_6', 'Type_Failure_6'), axis = 1)

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
    
k = ['4', '6', '8', '10', '12', '14', '16']
sq_comm = data.loc[data['CentralControl_actor'] == 'receiver']
sq_comm1 = sq_comm.loc[sq_comm['Type_Difference_5'] == True]
#print(sq_comm1['utility_5'].mean())
#print(sq_comm1['SigLength_5'].mean())
#print(sq_comm1['utility_5_baseline'].mean())
sq_commCount = sq_comm.loc[sq_comm['bits_0'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_1'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_2'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_3'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_4'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_5'] >0]
sq_commCount = sq_commCount.loc[sq_commCount['bits_6'] >0]

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

print(sq_comm['Type_Difference_4'].value_counts())
print(sq_comm['Type_Difference_5'].value_counts())
print(sq_comm['Type_Difference_6'].value_counts())


"""
acc0, accDF0, acc1, accDF1, acc2, accDF2, acc3, accDF3, acc4, accDF4, acc5, accDF5, acc6, accDF6 = facetByItem(sq_comm)

base = np.log2(12*8*4)
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

plt.ylim(0, 10)
#ax[1].xlabel('Signal Cost Per Word', fontsize = 15)
plt.legend()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Number of Items', fontsize = 25)
plt.ylabel('Average Bits', fontsize = 25)
sns.despine()
#plt.xlabel('Signal Cots Per Word')
#Set Quit or DIY Equlas to 0
plt.title ('Average Bits --- Quit or DIY = 0', fontsize = 25)
"""
"""
Graphsa to make Bit Average Across All Items
bits = getProportionTargetReached355(sq_commCount)['count_bits']
bits_std = getProportionTargetReached355(sq_commCount)['marginOfErrorStd']
#If remove all rows the contain Quit or DIY.
 # 1071 rows

g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6']
m = [np.log2(162), np.log2(162), np.log2(162), np.log2(162), np.log2(162), np.log2(162), np.log2(162) ]
plt.errorbar(g, bits, yerr = bits_std, label = 'Imagined We Model', color = '#ad0b2d', linewidth = 4 )
plt.errorbar(g, m, label = 'Baseline Model',  color = '#555555', linewidth = 4 )
plt.ylim(0, 10)
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
plt.title ('Average Bits --- Remove All Rows that Any Cost Choose to Quit or DIY', fontsize = 25)
"""
#plt.show()

