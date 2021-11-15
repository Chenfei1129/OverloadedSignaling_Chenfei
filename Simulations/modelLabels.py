
# Item and Feature Spaces
itemSpace_3Color3Shape = ['green triangle', 'purple triangle', 'red triangle', 'green circle', 'purple circle', 'red circle', 'green square', 'purple square', 'red square']
featureSpace_3Color3Shape = ['green', 'purple', 'red', 'circle', 'triangle', 'square']


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