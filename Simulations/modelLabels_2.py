
# Item and Feature Spaces
itemSpace_3Color3Shape = ['green triangle small', 'green triangle medium', 'green triangle large', 'green triangle smallest', 
                          'purple triangle small', 'purple triangle medium', 'purple triangle large','purple triangle smallest',
                          'red triangle small', 'red triangle medium', 'red triangle large', 'red triangle smallest',
                          'green circle small', 'green circle medium', 'green circle large', 'green circle smallest',
                          'purple circle small', 'purple circle medium', 'purple circle large', 'purple circle smallest',
                          'red circle small', 'red circle medium', 'red circle large', 'red circle smallest', 
                          'green square small', 'green square medium', 'green square large', 'green square smallest',
                          'purple square small', 'purple square medium', 'purple square large', 'purple square smallest', 
                          'red square small', 'red square medium', 'red square large', 'red square smallest',
                          'blue triangle small', 'blue triangle medium', 'blue triangle large', 'blue triangle smallest',
                          'blue circle small', 'blue circle medium', 'blue circle large', 'blue circle smallest',
                          'blue square small', 'blue square medium', 'blue square large', 'blue square smallest',
                          'green oval small', 'green oval medium', 'green oval large', 'green oval smallest', 
                          'purple oval small', 'purple oval medium', 'purple oval large','purple oval smallest',
                          'red oval small', 'red oval medium', 'red oval large', 'red oval smallest',
                          'blue oval small', 'blue oval medium', 'blue oval large', 'blue oval smallest'
                          ]
featureSpace_3Color3Shape = ['green', 'purple', 'red', 'blue', 'circle', 'triangle', 'square', 'oval', 'small', 'medium', 'large', 'smallest' ]


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
