# Item and Feature Spaces
itemSpace_3Color3Shape = ['green triangle small shaded', 'green triangle medium shaded', 'green triangle large shaded',
                          'purple triangle small shaded', 'purple triangle medium shaded', 'purple triangle large shaded',
                          'red triangle small shaded', 'red triangle medium shaded', 'red triangle large shaded', 
                          'green circle small shaded', 'green circle medium shaded', 'green circle large shaded', 
                          'purple circle small shaded', 'purple circle medium shaded', 'purple circle large shaded', 
                          'red circle small shaded', 'red circle medium shaded', 'red circle large shaded', 
                          'green square small shaded', 'green square medium shaded', 'green square large shaded', 
                          'purple square small shaded', 'purple square medium shaded', 'purple square large shaded', 
                          'red square small shaded', 'red square medium shaded', 'red square large shaded',                          
                          'green triangle small halfshaded', 'green triangle medium halfshaded', 'green triangle large halfshaded',
                          'purple triangle small halfshaded', 'purple triangle medium halfshaded', 'purple triangle large halfshaded',
                          'red triangle small halfshaded', 'red triangle medium halfshaded', 'red triangle large halfshaded', 
                          'green circle small halfshaded', 'green circle medium halfshaded', 'green circle large halfshaded', 
                          'purple circle small halfshaded', 'purple circle medium halfshaded', 'purple circle large halfshaded', 
                          'red circle small halfshaded', 'red circle medium halfshaded', 'red circle large halfshaded', 
                          'green square small halfshaded', 'green square medium halfshaded', 'green square large halfshaded', 
                          'purple square small halfshaded', 'purple square medium halfshaded', 'purple square large halfshaded', 
                          'red square small halfshaded', 'red square medium halfshaded', 'red square large halfshaded',                          
                          'green triangle small noshaded', 'green triangle medium noshaded', 'green triangle large noshaded',
                          'purple triangle small noshaded', 'purple triangle medium noshaded', 'purple triangle large noshaded',
                          'red triangle small noshaded', 'red triangle medium noshaded', 'red triangle large noshaded', 
                          'green circle small noshaded', 'green circle medium noshaded', 'green circle large noshaded', 
                          'purple circle small noshaded', 'purple circle medium noshaded', 'purple circle large noshaded', 
                          'red circle small noshaded', 'red circle medium noshaded', 'red circle large noshaded', 
                          'green square small noshaded', 'green square medium noshaded', 'green square large noshaded', 
                          'purple square small noshaded', 'purple square medium noshaded', 'purple square large noshaded', 
                          'red square small noshaded', 'red square medium noshaded', 'red square large noshaded'
                          ]
featureSpace_3Color3Shape = ['green', 'purple', 'red', 'circle', 'triangle', 'square', 'small', 'medium', 'large', 'noshaded', 'halfshaded', 'shaded' ]


# Trial Sampling Labels
S_LOCATION = 'signalerLocation'
R_LOCATION = 'receiverLocation'
INTENTION = 'intention'
SIGNAL_SPACE = 'signalSpace'
TARGET_DICT = 'targetDictionary'
SAMPLED_TRIAL_PARAMETERS = [S_LOCATION, R_LOCATION, INTENTION, SIGNAL_SPACE, TARGET_DICT, 'nTargets', 'propRelevantSignalsInVocab']


# Model Metrics Column Labels
SHARED_METRICS = ['sChoice', 'rChoice', 'sAchievesGoal', 'rAchievesGoal', 'goalAchieved', 'utility']
MODEL_NAMES = ['IW_N','IW_P','RSA_NL','RSA_PL','RSA_PA','JU']

MODELCOLUMNNAMESR0_IWnp_RSAnp_RSAAnp_JU = [
										'R0_IW_N_sChoice',
										'R0_IW_N_rChoice',
										'R0_IW_N_sAchievesGoal',
										'R0_IW_N_rAchievesGoal',
										'R0_IW_N_goalAchieved',
										'R0_IW_N_utility',
										'R0_IW_P_sChoice',
										'R0_IW_P_rChoice',
										'R0_IW_P_sAchievesGoal',
										'R0_IW_P_rAchievesGoal',
										'R0_IW_P_goalAchieved',
										'R0_IW_P_utility',
										'R0_RSA_NL_sChoice',
										'R0_RSA_NL_rChoice',
										'R0_RSA_NL_sAchievesGoal',
										'R0_RSA_NL_rAchievesGoal',
										'R0_RSA_NL_goalAchieved',
										'R0_RSA_NL_utility',
										'R0_RSA_PL_sChoice',
										'R0_RSA_PL_rChoice',
										'R0_RSA_PL_sAchievesGoal',
										'R0_RSA_PL_rAchievesGoal',
										'R0_RSA_PL_goalAchieved',
										'R0_RSA_PL_utility',
										'R0_RSA_PA_sChoice',
										'R0_RSA_PA_rChoice',
										'R0_RSA_PA_sAchievesGoal',
										'R0_RSA_PA_rAchievesGoal',
										'R0_RSA_PA_goalAchieved',
										'R0_RSA_PA_utility',
										'R0_RSA_NA_sChoice',
										'R0_RSA_NA_rChoice',
										'R0_RSA_NA_sAchievesGoal',
										'R0_RSA_NA_rAchievesGoal',
										'R0_RSA_NA_goalAchieved',
										'R0_RSA_NA_utility',
										'R0_JU_sChoice',
										'R0_JU_rChoice',
										'R0_JU_sAchievesGoal',
										'R0_JU_rAchievesGoal',
										'R0_JU_goalAchieved',
										'R0_JU_utility',
										'R0_CentralControl_utility',
										'R0_CentralControl_actor']




