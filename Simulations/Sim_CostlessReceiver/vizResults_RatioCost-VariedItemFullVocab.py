import pandas as pd
import numpy as np
import os
import sys	
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
#import visualizationsRE as viz
import matplotlib.pyplot as plt
from Simulations.visualizationsRE import getProportionTargetReached, getPercentFromOptimalUtilityDF, getUtilityDifferenceSummaryStatistics
sim_r0 = pd.read_pickle('./simulation_costRatioReceiver_a4_r8_seed423_2-9ItemFullVocabActual.pkl')

b1_comm = sim_r0.loc[sim_r0['CentralControl_actor'] == 'receiver']

print(getUtilityDifferenceSummaryStatistics(getPercentFromOptimalUtilityDF(b1_comm)))
#print(sim_r0.shape)#.iloc[0:5]
#print(pd.show_versions())
#gapminder.loc[:, gapminder.columns.str.endswith("1957")]
def getUtilityDifferenceSummaryStatistics(df):
    summaryDF = pd.DataFrame(df.mean(axis=0)).rename(columns={0:'means'})
    summaryDF['stds'] = df.std(axis=0)
    numObs = df.shape[0]
    summaryDF['marginOfError'] = 1.96*summaryDF['stds']/np.sqrt(numObs)
    summaryDF['CI_low'] = summaryDF['means'] - summaryDF['marginOfError']
    summaryDF['CI_high'] = summaryDF['means'] + summaryDF['marginOfError']
    return(summaryDF)
#print(getUtilityDifferenceSummaryStatistics(sim_r0))
plt.show()
