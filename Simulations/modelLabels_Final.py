itemSpace_3Color3Shape = []
colors = ['green', 'purple', 'red']
shapes = ['circle', 'triangle', 'square']
sizes = ['small', 'medium', 'large']
for color in colors:
    for shape in shapes:
        for size in sizes:
            itemSpace_3Color3Shape.append(color + ' ' + shape + ' ' + size)

featureSpace_3Color3Shape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large']


itemSpace_3Color4Shape = []
colors = ['green', 'purple', 'red', 'blue']
shapes = ['circle', 'triangle', 'square', 'oval']
sizes = ['small', 'medium', 'large', 'smallest']
for color in colors:
    for shape in shapes:
        for size in sizes:
            itemSpace_3Color4Shape.append(color + ' ' + shape + ' ' + size)

featureSpace_3Color4Shape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 
                             'blue','oval', 'smallest']


itemSpace_3Color5Shape = []
colors = ['green', 'purple', 'red', 'blue', 'orange']
shapes = ['circle', 'triangle', 'square', 'oval', 'star']
sizes = ['small', 'medium', 'large', 'smallest', 'largest']
for color in colors:
    for shape in shapes:
        for size in sizes:
            itemSpace_3Color5Shape.append(color + ' ' + shape + ' ' + size)

featureSpace_3Color5hape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 
                             'blue','orange','oval', 'star','smallest', 'largest']


itemSpace_4Color3Shape = []
colors = ['green', 'purple', 'red']
shapes = ['circle', 'triangle', 'square']
sizes = ['small', 'medium', 'large']
shades = ['noshade',  'halfshade' , 'shade']
for color in colors:
    for shape in shapes:
        for size in sizes:
            for shade in shades:
                itemSpace_4Color3Shape.append(color + ' ' + shape + ' ' + size + ' ' + shade)
            
featureSpace_4Color3hape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 'noshade',  'halfshade' , 'shade']


itemSpace_4Color4Shape = []
colors = ['green', 'purple', 'red', 'blue']
shapes = ['circle', 'triangle', 'square', 'oval']
sizes = ['small', 'medium', 'large', 'smallest']
shades = ['noshade',  'halfshade' , 'shade', 'quartershade']
for color in colors:
    for shape in shapes:
        for size in sizes:
            for shade in shades:
                itemSpace_4Color4Shape.append(color + ' ' + shape + ' ' + size + ' ' + shade)
            

featureSpace_4Color4hape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 'noshade',  'halfshade' , 'shade', 
                             'blue','oval', 'smallest', 'quartershade']

            

itemSpace_4Color5Shape = []
colors = ['green', 'purple', 'red', 'blue', 'orange']
shapes = ['circle', 'triangle', 'square', 'oval', 'star']
sizes = ['small', 'medium', 'large', 'smallest', 'largest']
shades = ['noshade',  'halfshade' , 'shade', 'quartershade', 'greyshade']
for color in colors:
    for shape in shapes:
        for size in sizes:
            for shade in shades:
                itemSpace_4Color5Shape.append(color + ' ' + shape + ' ' + size + ' ' + shade)
            
featureSpace_4Color5hape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 'noshade',  'halfshade' , 'shade', 
                             'blue','orange','oval', 'star','smallest', 'largest','quartershade', 'greyshade']


itemSpace_5Color3Shape = []
colors = ['green', 'purple', 'red']
shapes = ['circle', 'triangle', 'square']
sizes = ['small', 'medium', 'large']
shades = ['noshade',  'halfshade' , 'shade']
heights = ['h1',  'h2' , 'h3']
for color in colors:
    for shape in shapes:
        for size in sizes:
            for shade in shades:
                for height in heights:
                    itemSpace_5Color3Shape.append(color + ' ' + shape + ' ' + size + ' ' + shade + ' ' + height)
                
featureSpace_5Color3Shape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 
                             'noshade',  'halfshade' , 'shade', 
                             'h1',  'h2' , 'h3']


itemSpace_5Color4Shape = []
colors = ['green', 'purple', 'red', 'blue']
shapes = ['circle', 'triangle', 'square', 'oval']
sizes = ['small', 'medium', 'large', 'smallest']
shades = ['noshade',  'halfshade' , 'shade', 'quartershade']
heights = ['h1',  'h2' , 'h3', 'h4']
for color in colors:
    for shape in shapes:
        for size in sizes:
            for shade in shades:
                for height in heights:
                    itemSpace_5Color4Shape.append(color + ' ' + shape + ' ' + size + ' ' + shade + ' ' + height)
                
featureSpace_5Color4Shape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 
                             'noshade',  'halfshade' , 'shade', 
                             'h1',  'h2' , 'h3', 'blue','oval', 'smallest', 'quartershade', 'h4']


itemSpace_5Color5Shape = []
colors = ['green', 'purple', 'red', 'blue', 'orange']
shapes = ['circle', 'triangle', 'square', 'oval', 'star']
sizes = ['small', 'medium', 'large', 'smallest', 'largest']
shades = ['noshade',  'halfshade' , 'shade', 'quartershade', 'greyshade']
heights = ['h1',  'h2' , 'h3', 'h4', 'h5']
for color in colors:
    for shape in shapes:
        for size in sizes:
            for shade in shades:
                for height in heights:
                    itemSpace_5Color5Shape.append(color + ' ' + shape + ' ' + size + ' ' + shade + ' ' + height)
                

featureSpace_5Color5Shape = ['green', 'purple', 'red', 
                             'circle', 'triangle', 'square', 
                             'small', 'medium', 'large', 
                             'noshade',  'halfshade' , 'shade', 
                             'h1',  'h2' , 'h3', 'blue','orange','oval', 'star','smallest', 'largest','quartershade', 'greyshade','h4', 'h5']




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

                
