import pandas as pd
import numpy as np
import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
#import visualizationsRE as viz
import matplotlib.pyplot as plt
from Simulations.visualizationsRE import getProportionTargetReached, getPercentFromOptimalUtilityDF, getUtilityDifferenceSummaryStatistics, facetAccuracyByNItem, plotStackedCommunicationByItem
sim_r0 = pd.read_pickle('./simulation_costRaioReceiver_a4_r8_seed423_2-9ItemFullVocab_2000_IW_signaler_report.pkl')
sq_comm = sim_r0.loc[sim_r0['CentralControl_actor'] == 'receiver']
accDict = facetAccuracyByNItem(sq_comm)
print(accDict)
totalAcc = pd.DataFrame(columns = ['IW_S1R0_sAchievesGoal','IW_S1R0_sAchievesGoalC_R0.2' ,'IW_S1R0_sAchievesGoalC_R0.4' , 'IW_S1R0_sAchievesGoalC_R0.6' ,  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1' ])

totalAcc_ME = pd.DataFrame(columns = ['IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1' ] )

successComm = pd.DataFrame(columns = ['IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1'  ] )
successComm_ME = pd.DataFrame(columns = [ 'IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1'  ] )

for nItem, accDF in accDict.items():


    #totalAcc.loc[nItem] = accDF['total'].loc[['JU_sAchievesGoal', 'JU_sAchievesGoalC_R0.2', 'JU_sAchievesGoalC_R0.4', 'JU_sAchievesGoalC_R0.6',  'JU_sAchievesGoalC_R0.8', 'JU_sAchievesGoalC_R1' ]].values
    totalAcc.loc[nItem] = accDF['total'].loc[['IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1'  ]].values
    totalAcc_ME.loc[nItem] = accDF['marginOfErrorT'].loc[['IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1'   ]].values
    
    successComm.loc[nItem] = accDF['receiver'].loc[['IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1'   ]].values
    successComm_ME.loc[nItem] = accDF['marginOfErrorR'].loc[['IW_S1R0_sAchievesGoal' ,'IW_S1R0_sAchievesGoalC_R0.2', 'IW_S1R0_sAchievesGoalC_R0.4', 'IW_S1R0_sAchievesGoalC_R0.6',  'IW_S1R0_sAchievesGoalC_R0.8', 'IW_S1R0_sAchievesGoalC_R1'  ]].values

totalAcc = totalAcc.reindex(["2", "3", "4", '5', '6', '7', '8', '9'])
totalAcc_ME = totalAcc_ME.reindex(["2", "3", "4", '5', '6', '7', '8', '9'])

successComm = successComm.reindex(["2", "3", "4", '5', '6', '7', '8', '9'])
successComm_ME = successComm_ME.reindex(["2", "3", "4", '5', '6', '7', '8', '9'])
print(successComm)
def plotMetricByItem3(accDf, meDF, save = False, filename = './acc.png', yAxisLabel = 'Proportion successful communication', yTicks = [0,.1, .2, .3, .4, .5,.6, .7, .8, .9, 1], fSize= (8,5),minItems = 2, maxItems = 9):
    fig = plt.figure(figsize =fSize)
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    labelDict = {'IW_S1R0_sAchievesGoal':'Receiver Cost Ratio 0',
                 'IW_S1R0_sAchievesGoalC_R0.2':'Receiver Cost Ratio 0.2',
                 'IW_S1R0_sAchievesGoalC_R0.4':'Receiver Cost Ratio 0.4',
                 'IW_S1R0_sAchievesGoalC_R0.6':'Receiver Cost Ratio 0.6',
                 'IW_S1R0_sAchievesGoalC_R0.8':'Receiver Cost Ratio 0.8',
                 'IW_S1R0_sAchievesGoalC_R1':'Receiver Cost Ratio 1'}

    colorDict = {'IW_S1R0_sAchievesGoalC_R1': 'orange',
                 'IW_S1R0_sAchievesGoalC_R0.8': '#b10d2f', 
                 'IW_S1R0_sAchievesGoalC_R0.6': '#95a84c',
                 'IW_S1R0_sAchievesGoalC_R0.4': "#91afec", 
                 'IW_S1R0_sAchievesGoalC_R0.2': '#555555',
                 'IW_S1R0_sAchievesGoal': '#feb954'}

    modelLabels = accDf.columns

    x = [a for a in range(minItems, maxItems+1)]

    for modelLabel in modelLabels:
        plt.errorbar(x, accDf[modelLabel].values, yerr = meDF[modelLabel].values, color = colorDict[modelLabel],label=labelDict[modelLabel], linewidth=3)
    plt.xlabel('Number of Items')
    plt.ylabel(yAxisLabel)
    plt.legend(loc='best')
    ax.set_yticks(yTicks)
    ax.set_title('IW Model S1R0')
    ax.tick_params(labelsize=24)
    if save:
        fig.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()

plotMetricByItem3(successComm, successComm_ME,
                     save = False, 
                     filename = './acc.png', 
                     yAxisLabel = 'Proportion successful communication', 
                     yTicks = [1,.75,.50,.25,0], 
                     fSize= (8,5),minItems = 2, maxItems = 9)

'''
plotStackedCommunicationByItem(ax[0],
                               accuracyDict= accDict, 
                               modelToPlot='IW_S1R1_sAchievesGoal', 
                               modelName='Imagined We C_R 0',
                               sCanQuit = False,
                               showLegend = True, 
                               succCommColor = "#000650", failCommColor = '#9FA5BD', doColor = '#f8d7c0',quitColor='#f58960'
                              )
plotStackedCommunicationByItem(ax[1],
                               accuracyDict= accDict, 
                               modelToPlot='IW_S1R1_sAchievesGoalC_R0.2', 
                               modelName='Imagined We C_R 0.2',
                               sCanQuit = False,
                               showLegend = True, 
                               succCommColor = "#000650", failCommColor = '#9FA5BD', doColor = '#f8d7c0',quitColor='#f58960'
                              )

plotStackedCommunicationByItem(ax[2],
                               accuracyDict= accDict, 
                               modelToPlot='IW_S1R1_sAchievesGoalC_R0.4', #C_R1
                               modelName='Imagined We C_R 0.4',
                               sCanQuit = False,
                               showLegend = True, 
                               succCommColor = "#000650", failCommColor = '#9FA5BD', doColor = '#f8d7c0',quitColor='#f58960'
                              )

plotStackedCommunicationByItem(ax[3],
                               accuracyDict= accDict, 
                               modelToPlot='IW_S1R1_sAchievesGoalC_R0.6', #C_R1
                               modelName='Imagined We C_R 0.6',
                               sCanQuit = False,
                               showLegend = True, 
                               succCommColor = "#000650", failCommColor = '#9FA5BD', doColor = '#f8d7c0',quitColor='#f58960'
                              )

plotStackedCommunicationByItem(ax[4],
                               accuracyDict= accDict, 
                               modelToPlot='IW_S1R1_sAchievesGoalC_R0.8', #C_R1
                               modelName='Imagined We C_R 0.8',
                               sCanQuit = False,
                               showLegend = True, 
                               succCommColor = "#000650", failCommColor = '#9FA5BD', doColor = '#f8d7c0',quitColor='#f58960'
                              )

plotStackedCommunicationByItem(ax[5],
                               accuracyDict= accDict, 
                               modelToPlot='IW_S1R1_sAchievesGoalC_R1', #C_R1
                               modelName='Imagined We C_R 1',
                               sCanQuit = False,
                               showLegend = True, 
                               succCommColor = "#000650", failCommColor = '#9FA5BD', doColor = '#f8d7c0',quitColor='#f58960'
                              )

plt.show()

"""

nTargets                       5.82  2.341458  ...  5.361074  6.278926
propRelevantSignalsInVocab     1.00  0.000000  ...  1.000000  1.000000
IW_S1R1_sAchievesGoal          0.53  0.501614  ...  0.431684  0.628316
IW_S1R1_rAchievesGoal          0.07  0.256432  ...  0.019739  0.120261
IW_S1R1_goalAchieved           0.60  0.492366  ...  0.503496  0.696504
IW_S1R1_utility                2.45  2.602932  ...  1.939825  2.960175
CentralControl_utility
'''
