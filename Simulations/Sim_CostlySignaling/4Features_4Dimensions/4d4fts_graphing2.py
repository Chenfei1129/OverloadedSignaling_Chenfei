import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
from processCostlySignalingSimulation import findNumberofWordsinBaseline, getBaselineActionChoice, getSignalLength, calculateBits, getUtilityWithSignalCost, baselineChoice, getTypeOfReceiverFailure, isDIYBetterThanCCWithCosts, getUtilityWithSignalCost,getUtilityWithSignalCost_base 
import matplotlib.pyplot as plt
data = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_0.2_0.4_costs_round1.pkl')
data2 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_0.2_0.4_costs_round2.pkl')
data3 = pd.read_pickle('./simulation_4_dimensions_4_features_central_controller_0.2_0.4_costs_round3.pkl')
data = data.append(data2)
data = data.append(data3)
print(data.shape)

data_0_1 = pd.read_pickle('./dim4ft4_challengingEnv-sigQuit-aMax-R8-Seed282_Cost0_01_2000runs_central_controller.pkl')
#print(data.columns)
#print(data_0_1.columns)
data = data.rename({'signalerLocation':'signalerLocationS_C0.2', 'receiverLocation':'receiverLocationS_C0.2', 'intention':'intentionS_C0.2',
                   'signalSpace':'signalSpaceS_C0.2',
       'targetDictionary':'targetDictionaryS_C0.2', 'nTargets':'nTargetsS_C0.2',
       'IW_S1R0_sChoice':'IW_S1R0_sChoiceS_C0.2' , 'IW_S1R0_rChoice':'IW_S1R0_rChoiceS_C0.2', 'IW_S1R0_sAchievesGoal':'IW_S1R0_sAchievesGoalS_C0.2',
       'IW_S1R0_rAchievesGoal':'IW_S1R0_rAchievesGoalS_C0.2', 'IW_S1R0_goalAchieved':'IW_S1R0_goalAchievedS_C0.2',
       'IW_S1R0_utility':'IW_S1R0_utilityS_C0.2'
        }, axis = 1)

print(data_0_1.shape)
#data = pd.concat([data, data_0_1], axis = 1)
#print(data.shape)
#print(data.columns)
data['baseline'] = data.apply(lambda row:  4, axis = 1)
data_0_1['baseline'] = data_0_1.apply(lambda row:  4, axis = 1)
data_0_1['DIY_better_0'] = data_0_1.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0, 'baseline'), axis = 1)
data_0_1['DIY_better_1'] = data_0_1.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.1, 'baseline'), axis = 1)
data['DIY_better_2'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.2, 'baseline'), axis = 1)
data['DIY_better_3'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.3, 'baseline'), axis = 1)
data['DIY_better_4'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.4, 'baseline'), axis = 1)
data['DIY_better_5'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.5, 'baseline'), axis = 1)
data['DIY_better_6'] = data.apply(lambda row: isDIYBetterThanCCWithCosts(row, 0.6, 'baseline'), axis = 1)

data_0_1['Type_Failure_0'] = data_0_1.apply(lambda row: getTypeOfReceiverFailure(row), axis = 1)
data_0_1['Type_Failure_1'] = data_0_1.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.1', 'DIY_better_1', 'IW_S1R0_goalAchievedS_C0.1'), axis = 1)
data['Type_Failure_2'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.2', 'DIY_better_2', 'IW_S1R0_goalAchievedS_C0.2'), axis = 1)
data['Type_Failure_3'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.3', 'DIY_better_3', 'IW_S1R0_goalAchievedS_C0.3'), axis = 1)
data['Type_Failure_4'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.4', 'DIY_better_4', 'IW_S1R0_goalAchievedS_C0.4'), axis = 1)
data['Type_Failure_5'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.5', 'DIY_better_5', 'IW_S1R0_goalAchievedS_C0.5'), axis = 1)
data['Type_Failure_6'] = data.apply(lambda row: getTypeOfReceiverFailure(row, 'IW_S1R0_sChoiceS_C0.6', 'DIY_better_6', 'IW_S1R0_goalAchievedS_C0.6'), axis = 1)




data_0_1['baseline_0'] = data_0_1.apply(lambda row: baselineChoice(row, 0, 'baseline'), axis = 1)
data_0_1['baseline_1'] = data_0_1.apply(lambda row: baselineChoice(row, 0.1, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.1', diy_columnName='DIYSignaler_utilityS_C0.1'), axis = 1)
data['baseline_2'] = data.apply(lambda row: baselineChoice(row, 0.2, 'baseline', cc_columnName = 'CentralControl_utility', diy_columnName='DIYSignaler_utility'), axis = 1)
data['baseline_3'] = data.apply(lambda row: baselineChoice(row, 0.3, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.3', diy_columnName='DIYSignaler_utilityS_C0.3'), axis = 1)
data['baseline_4'] = data.apply(lambda row: baselineChoice(row, 0.4, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.4', diy_columnName='DIYSignaler_utilityS_C0.4'), axis = 1)
data['baseline_5'] = data.apply(lambda row: baselineChoice(row, 0.5, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.5', diy_columnName='DIYSignaler_utilityS_C0.5'), axis = 1)
data['baseline_6'] = data.apply(lambda row: baselineChoice(row, 0.6, 'baseline', cc_columnName = 'CentralControl_utilityS_C0.6', diy_columnName='DIYSignaler_utilityS_C0.6'), axis = 1)

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
             
             'quit':[ getPropSignalerQuits('Type_Failure_2'),
                     getPropSignalerQuits('Type_Failure_3'),getPropSignalerQuits('Type_Failure_4'), getPropSignalerQuits('Type_Failure_5'),getPropSignalerQuits('Type_Failure_6')
                     ],
             'DIY_Optimal':[getPropSignalerDIY_Optimal('Type_Failure_2'),
                     getPropSignalerDIY_Optimal('Type_Failure_3'),getPropSignalerDIY_Optimal('Type_Failure_4'), getPropSignalerDIY_Optimal('Type_Failure_5'),getPropSignalerDIY_Optimal('Type_Failure_6')
                     ],
             'DIY_subOptimal':[ getPropSignalerDIY_subOptimal('Type_Failure_2'),
                     getPropSignalerDIY_subOptimal('Type_Failure_3'),getPropSignalerDIY_subOptimal('Type_Failure_4'), getPropSignalerDIY_subOptimal('Type_Failure_5'),getPropSignalerDIY_subOptimal('Type_Failure_6')
                     ],
             'Communication_failure':[getPropSignalerCommunication_Fail('Type_Failure_2'),
                     getPropSignalerCommunication_Fail('Type_Failure_3'),getPropSignalerCommunication_Fail('Type_Failure_4'), getPropSignalerCommunication_Fail('Type_Failure_5'),getPropSignalerCommunication_Fail('Type_Failure_6')
                     ],
             'Communication':[getPropFalse('Type_Failure_2'),
                     getPropFalse('Type_Failure_3'),getPropFalse('Type_Failure_4'), getPropFalse('Type_Failure_5'),getPropFalse('Type_Failure_6')
                     ],
             'Communication_baseline':[ getPropCom('baseline_2'),
                     getPropCom('baseline_3'),getPropCom('baseline_4'), getPropCom('baseline_5'),getPropCom('baseline_6')
                     ],
             'quit_baseline':[ getPropSignalerQuits('baseline_2'),
                     getPropSignalerQuits('baseline_3'),getPropSignalerQuits('baseline_4'), getPropSignalerQuits('baseline_5'),getPropSignalerQuits('baseline_6')
                     ],
             'DIY_baseline':[ getPropDIY('baseline_2'),
                     getPropDIY('baseline_3'),getPropDIY('baseline_4'), getPropDIY('baseline_5'),getPropDIY('baseline_6')
                     ]
             
             }
    propTrialsTargetReached = pd.DataFrame(dfNew, columns = ['quit', 'DIY_Optimal', 'DIY_subOptimal', 'Communication_failure', 'Communication', 'Communication_baseline', 'quit_baseline',
                                                             'DIY_baseline'
                                                             ], index=['IW_S1R0S_C0.2', 'IW_S1R0S_C0.3','IW_S1R0S_C0.4', 'IW_S1R0S_C0.5','IW_S1R0S_C0.6' ])

    #propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd'] =  1.96*np.sqrt(propTrialsTargetReached['count_bits_std'] * propTrialsTargetReached['count_bits_std'] /df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd_utility_percent_base'] =  1.96*np.sqrt(propTrialsTargetReached['utility_base_percent_std' ] * propTrialsTargetReached['utility_base_percent_std' ] /df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd_utility_percent'] =  1.96*np.sqrt(propTrialsTargetReached['utility_percent_std' ] * propTrialsTargetReached['utility_percent_std'] /df.shape[0])
    #propTrialsTargetReached['receiver failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler'])
    #propTrialsTargetReached['receiver_failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler_does'])
    return propTrialsTargetReached

def getProportionTargetReached_01(df):
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0
    getPropFalse =  lambda colName: df[colName].value_counts(normalize=True).loc[False] if (False in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerQuits =  lambda colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerDIY_Optimal =  lambda colName: df[colName].value_counts(normalize=True).loc['do_DIY_Optimal'] if ('do_DIY_Optimal' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerDIY_subOptimal =  lambda colName: df[colName].value_counts(normalize=True).loc['do_DIY_suboptimal'] if ('do_DIY_suboptimal' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerCommunication_Fail =  lambda colName: df[colName].value_counts(normalize=True).loc['communication_failure'] if ('communication_failure' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropCom = lambda colName: df[colName].value_counts(normalize=True).loc['Communicate'] if ('Communicate' in df[colName].value_counts(normalize=True).index) else 0.0
    getPropDIY = lambda colName: df[colName].value_counts(normalize=True).loc['DIY'] if ('DIY' in df[colName].value_counts(normalize=True).index) else 0.0
    dfNew = {
             
             'quit':[getPropSignalerQuits('Type_Failure_0'),getPropSignalerQuits('Type_Failure_1')
                     ],
             'DIY_Optimal':[getPropSignalerDIY_Optimal('Type_Failure_0'),getPropSignalerDIY_Optimal('Type_Failure_1')
                     ],
             'DIY_subOptimal':[getPropSignalerDIY_subOptimal('Type_Failure_0'),getPropSignalerDIY_subOptimal('Type_Failure_1')
                     ],
             'Communication_failure':[getPropSignalerCommunication_Fail('Type_Failure_0'),getPropSignalerCommunication_Fail('Type_Failure_1')
                     ],
             'Communication':[getPropFalse('Type_Failure_0'),getPropFalse('Type_Failure_1')
                     ],
             'Communication_baseline':[getPropCom('baseline_0'),getPropCom('baseline_1')
                     ],
             'quit_baseline':[getPropSignalerQuits('baseline_0'),getPropSignalerQuits('baseline_1')
                     ],
             'DIY_baseline':[getPropDIY('baseline_0'),getPropDIY('baseline_1')
                     ]
             
             }
    propTrialsTargetReached = pd.DataFrame(dfNew, columns = ['quit', 'DIY_Optimal', 'DIY_subOptimal', 'Communication_failure', 'Communication', 'Communication_baseline', 'quit_baseline',
                                                             'DIY_baseline'
                                                             ], index=['IW_S1R0', 'IW_S1R0S_C0.1' ])

    #propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd'] =  1.96*np.sqrt(propTrialsTargetReached['count_bits_std'] * propTrialsTargetReached['count_bits_std'] /df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd_utility_percent_base'] =  1.96*np.sqrt(propTrialsTargetReached['utility_base_percent_std' ] * propTrialsTargetReached['utility_base_percent_std' ] /df.shape[0])
    #propTrialsTargetReached['marginOfErrorStd_utility_percent'] =  1.96*np.sqrt(propTrialsTargetReached['utility_percent_std' ] * propTrialsTargetReached['utility_percent_std'] /df.shape[0])
    #propTrialsTargetReached['receiver failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler'])
    #propTrialsTargetReached['receiver_failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler_does'])
    return propTrialsTargetReached
quit_total = getProportionTargetReached_01(data_0_1)['quit_baseline'] * 100
quit_total = quit_total.append(getProportionTargetReached(data)['quit_baseline'] * 100)
quit_IW = getProportionTargetReached_01(data_0_1)['quit'] * 100
quit_IW = quit_IW.append(getProportionTargetReached(data)['quit'] * 100)
DIY = getProportionTargetReached_01(data_0_1)['DIY_baseline'] * 100
DIY = DIY.append(getProportionTargetReached(data)['DIY_baseline'] * 100)
DIY_large = getProportionTargetReached_01(data_0_1)['DIY_Optimal']*100
DIY_large = DIY_large.append(getProportionTargetReached(data)['DIY_Optimal']*100)
receiver_failed = getProportionTargetReached_01(data_0_1)['Communication_failure'] * 100
receiver_failed = receiver_failed.append(getProportionTargetReached(data)['Communication_failure'] * 100)
#print(getProportionTargetReached(sq_comm)['Communication'])
print(quit_total)
print(DIY)
print(DIY_large)
print(quit_IW)
import seaborn as sns

#g = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
g = np.arange(7) 
width = 0.35
fig = plt.figure(figsize = (16, 5))
fig, ax = plt.subplots()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#plt.bar(g, DIY_large,label = 'Signaler Does for Self(suboptimal)', edgecolor='white'  )
"""
ax.bar(g-width/2, quit_total, width, label = 'Signaler Quits --- Baseline', edgecolor = 'white', color = '#000650')
ax.bar(g-width/2, DIY, width, label ='Signaler Does for Self(perferred given signal cost) --- Baseline',edgecolor='white', bottom = quit_total, color = '#91AFEC' )
ax.bar(g+width/2, quit_IW, width, label = 'Signaler Quits --- IW', edgecolor = 'white', color = '#000650', alpha = 0.5)
ax.bar(g+width/2, DIY_large, width, label ='Signaler Does for Self(perferred given signal cost) --- IW',edgecolor='white', bottom = quit_IW, color = '#91AFEC', alpha = 0.5 )
"""
#plt.bar(g, receiver_failed, label = 'Communication Fails')
ax.bar(g-width/2, quit_total, width, label = 'Signaler Quits --- Baseline', edgecolor = 'white', color = '#000650', alpha = 0.5)
ax.bar(g-width/2, DIY, width, label ='Signaler Does for Self(perferred given signal cost) --- Baseline',edgecolor='white', bottom = quit_total, color = '#91AFEC', alpha = 0.5)
ax.bar(g+width/2, quit_IW, width, label = 'Signaler Quits --- IW', edgecolor = 'white', color = '#000650')
ax.bar(g+width/2, DIY_large, width, label ='Signaler Does for Self(perferred given signal cost) --- IW',edgecolor='white', bottom = quit_IW, color = '#91AFEC')

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
