import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def getProportionTargetReached(df):
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0
    getPropSignalerQuits =  lambda colName: df[colName].value_counts(normalize=True).loc['quit'] if ('quit' in df[colName].value_counts(normalize=True).index) else 0.0  
    dfNew = {'signaler': [getPropTrue('IW_S1R0_sAchievesGoal'), getPropTrue('IW_S1R0_sAchievesGoalS_C0.2') , getPropTrue('IW_S1R0_sAchievesGoalS_C0.3'), getPropTrue('IW_S1R0_sAchievesGoalS_C0.5')] ,
             'signaler_does': [getPropTrue('do_1'), getPropTrue('do_2'), getPropTrue('do_3'), getPropTrue('do_5')] ,
             'receiver': [getPropTrue('IW_S1R0_rAchievesGoal'),getPropTrue('IW_S1R0_rAchievesGoalS_C0.2'), getPropTrue('IW_S1R0_rAchievesGoalS_C0.3'),getPropTrue('IW_S1R0_rAchievesGoalS_C0.5')],
             
             'receiver_DIY_larger':[getPropTrue('larger_0.1_F'), getPropTrue('larger_0.2_F' ), getPropTrue('larger_0.3_F'), getPropTrue('larger_0.5_F' )],
             
             'receiver_quit_larger':[getPropTrue('larger_0.1_F2'),getPropTrue('larger_0.2_F2' ), getPropTrue('larger_0.3_F2'),getPropTrue('larger_0.5_F2' )],
             
              'receiver_fail_larger':[getPropTrue('larger_0.1_F22'), getPropTrue('larger_0.2_F22' ), getPropTrue('larger_0.3_F22'), getPropTrue('larger_0.5_F22' )],
             
             'receiver_1word': [getPropTrue('signal_length_1word_IW_S1R0_rAchievesGoalS_C0.1'), 
                                getPropTrue('signal_length_1word_IW_S1R0_rAchievesGoalS_C0.2'),
                                getPropTrue('signal_length_1word_IW_S1R0_rAchievesGoalS_C0.3'), 
                                getPropTrue('signal_length_1word_IW_S1R0_rAchievesGoalS_C0.5')], 
             
             
             'receiver_2word': [getPropTrue('signal_length_2word_IW_S1R0_rAchievesGoalS_C0.1'),
                                getPropTrue('signal_length_2word_IW_S1R0_rAchievesGoalS_C0.2'),
                                getPropTrue('signal_length_2word_IW_S1R0_rAchievesGoalS_C0.3'),
                                getPropTrue('signal_length_2word_IW_S1R0_rAchievesGoalS_C0.5')],
             
             'receiver_3word': [getPropTrue('signal_length_3word_IW_S1R0_rAchievesGoalS_C0.1'),
                                getPropTrue('signal_length_3word_IW_S1R0_rAchievesGoalS_C0.2'),
                                getPropTrue('signal_length_3word_IW_S1R0_rAchievesGoalS_C0.3'),
                                getPropTrue('signal_length_3word_IW_S1R0_rAchievesGoalS_C0.5')],
             
             'quit':[getPropSignalerQuits('IW_S1R0_sChoice'),getPropSignalerQuits('IW_S1R0_sChoiceS_C0.2'),
                     getPropSignalerQuits('IW_S1R0_sChoiceS_C0.3'),getPropSignalerQuits('IW_S1R0_sChoiceS_C0.5')], 

             'count': [df['countS_C0.1'].mean(), df['countS_C0.2'].mean(), df['countS_C0.3'].mean(), df['countS_C0.5'].mean() ], 
             'count_std': [df['countS_C0.1'].std(), df['countS_C0.2'].std(), df['countS_C0.3'].std(), df['countS_C0.5'].std()],
             'count_base' : [df['inOrNot_count'].mean(), df['inOrNot_count'].mean(), df['inOrNot_count'].mean(), df['inOrNot_count'].mean()],
             'count_base_std' : [df['inOrNot_count'].std(), df['inOrNot_count'].std(),df['inOrNot_count'].std(), df['inOrNot_count'].std() ],
             'utility': [df['utility_1'].mean()/df['CentralControl_utility'].mean(), df['utility_2'].mean()/df['CentralControl_utility'].mean(),df['utility_3'].mean()/df['CentralControl_utility'].mean(), df['utility_5'].mean()/df['CentralControl_utility'].mean() ]
             
             }
    propTrialsTargetReached = pd.DataFrame(dfNew, columns = ['count', 'count_std', 'count_base', 'count_base_std', 'signaler_does', 
                                                             'signaler', 'receiver', 'receiver_DIY_larger','receiver_quit_larger', 'receiver_fail_larger', 'utility',
                                                             'receiver_1word',  'receiver_2word', 'receiver_3word', 'quit'], index=['IW_S1R0S_C0.1' , 'IW_S1R0S_C0.2' , 'IW_S1R0S_C0.3' ,
                                                                    'IW_S1R0S_C0.5'])

    propTrialsTargetReached['unsuccessful communication'] = 1-(propTrialsTargetReached['receiver'])
    propTrialsTargetReached['receiver_DIY_small'] = propTrialsTargetReached['signaler_does'] -(propTrialsTargetReached['receiver_DIY_larger'])
    propTrialsTargetReached['receiver_quit_small'] = propTrialsTargetReached['quit'] -(propTrialsTargetReached['receiver_quit_larger'])
    propTrialsTargetReached['receiver_fail_small'] = propTrialsTargetReached['unsuccessful communication'] - propTrialsTargetReached['signaler_does'] - propTrialsTargetReached['quit'] - propTrialsTargetReached['receiver_fail_larger']
    
    propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorStd'] =  1.96*np.sqrt(propTrialsTargetReached['count_std'] * propTrialsTargetReached['count_std'] /df.shape[0])
    propTrialsTargetReached['marginOfErrorStd_base'] =  1.96*np.sqrt(propTrialsTargetReached['count_base_std'] * propTrialsTargetReached['count_base_std'] /df.shape[0])    
    propTrialsTargetReached['receiver failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler'])
    propTrialsTargetReached['receiver_failure'] = 1-(propTrialsTargetReached['receiver']+propTrialsTargetReached['quit']+ propTrialsTargetReached['signaler_does'])
    return propTrialsTargetReached
