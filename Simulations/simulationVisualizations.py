import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class ProcessSimulationData():
	def __init__(self, subsetByItem=False, subsetToCommunicationOptimal=True, utilPercent = True):
		self.subsetByItem = subsetByItem
		self.subsetToCommunicationOptimal = subsetToCommunicationOptimal
		self.utilPercent = utilPercent

		self.modelNames = ['IW Naive', 'IW Pragmatic','RSA Naive Language','RSA Pragmatic Language','RSA Pragmatic Action','Joint Utility']

		
		self.modelDict = {'IW Naive': {'util': 'IW_N_utility', 'sGoal': 'IW_N_sAchievesGoal', 'rGoal': 'IW_N_rAchievesGoal'}, 
								'IW Pragmatic': {'util': 'IW_P_utility', 'sGoal': 'IW_P_sAchievesGoal', 'rGoal': 'IW_P_rAchievesGoal'}, 
								'RSA Naive Language': {'util': 'RSA_NL_utility', 'sGoal': 'RSA_NL_sAchievesGoal', 'rGoal': 'RSA_NL_rAchievesGoal'}, 
								'RSA Pragmatic Language': {'util': 'RSA_PL_utility', 'sGoal': 'RSA_PL_sAchievesGoal', 'rGoal': 'RSA_PL_rAchievesGoal'}, 
								'RSA Pragmatic Action': {'util': 'RSA_PA_utility', 'sGoal': 'RSA_PA_sAchievesGoal', 'rGoal': 'RSA_PA_rAchievesGoal'}, 
								'Joint Utility': {'util': 'JU_utility', 'sGoal': 'JU_sAchievesGoal', 'rGoal': 'JU_rAchievesGoal'}}
		
	def __call__(self, simulationDF):
		dfForAnalysis = simulationDF.copy()
		if self.subsetToCommunicationOptimal:
			dfForAnalysis = simulationDF.loc[simulationDF['CentralControl_actor'] == 'receiver']

		if self.subsetByItem:
			summaries = self.facetAccuracyByNItem(dfForAnalysis)
		else:
			if self.utilPercent:
				utilDF = self.getPercentFromOptimalUtilityDF(dfForAnalysis)
			else:
				utilDF = self.getDifferenceFromOptimalUtilityDF(dfForAnalysis)

			utilSummary = self.getUtilityDifferenceSummaryStatistics(utilDF)
			accSummary = self.getProportionTargetReached(dfForAnalysis)
			summaries = [utilSummary, accSummary]

		return(summaries)

	def getDifferenceFromOptimalUtilityDF(self, df):
		r0_utilityLabel = 'R0_CentralControl_utility'
		CC_utilLabel = 'CentralControl_utility'
		differenceInUtility = pd.DataFrame(index=df.index)
		for modelName, modelDict in self.modelDict.items():
			if 'R0' in modelName:
				lab = r0_utilityLabel
			else:
				lab = CC_utilLabel

			utilLabel = modelDict['util']
			differenceInUtility[modelName] = df[lab]-df[utilLabel]
		differenceInUtility['DIYSignaler'] = df[lab]-df['DIYUtility']
		return(differenceInUtility)

	def getPercentFromOptimalUtilityDF(self, df):
		percentFromOptimalUtil = pd.DataFrame(index=df.index)
		for modelName, modelDict in self.modelDict.items():
			if 'R0' in modelName:
				u_optimal = 'R0_CentralControl_utility'
			else:
				u_optimal = 'CentralControl_utility'
			utilAchieved = modelDict['util']
			percentFromOptimalUtil[modelName] = 100.0*(df[utilAchieved]/df[u_optimal])
		percentFromOptimalUtil['DIYSignaler'] = 100.0*(df['DIYUtility']/df[u_optimal])
		return(percentFromOptimalUtil)

	def getUtilityDifferenceSummaryStatistics(self, df):
		summaryDF = pd.DataFrame(df.mean(axis=0)).rename(columns={0:'means'})
		summaryDF['stds'] = df.std(axis=0)
		numObs = df.shape[0]
		summaryDF['marginOfError'] = 1.96*summaryDF['stds']/np.sqrt(numObs)
		summaryDF['CI_low'] = summaryDF['means'] - summaryDF['marginOfError']
		summaryDF['CI_high'] = summaryDF['means'] + summaryDF['marginOfError']
		return(summaryDF)

	def getProportionTargetReached(self, df):
		propTrialsTargetReached = pd.DataFrame(columns=['signaler', 'receiver'])
		getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0

		for modelName, modelDict in self.modelDict.items():
			sLabel = modelDict['sGoal']
			rLabel = modelDict['rGoal']
			propTrialsTargetReached.loc[modelName] = [getPropTrue(sLabel), getPropTrue(rLabel)]

		propTrialsTargetReached['total'] = propTrialsTargetReached['signaler'] + propTrialsTargetReached['receiver']

		propTrialsTargetReached['marginOfErrorS'] =  1.96*np.sqrt((propTrialsTargetReached['signaler']*(1-propTrialsTargetReached['signaler']))/df.shape[0])
		propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])

		return(propTrialsTargetReached)

	def facetAccuracyByNItem(self, df):
		dfItems = df.copy()
		dlen = lambda x: len(x['targetDictionary'])
		dfItems['nTargets'] = dfItems.apply(dlen, axis=1) # gets the number of items in the target Dictionary
		
		accuracy = pd.DataFrame(index=self.modelNames)
		accuracyME = pd.DataFrame(index=self.modelNames)

		utility = pd.DataFrame(index = self.modelNames +['DIYSignaler'])
		utilityME = pd.DataFrame(index=self.modelNames+['DIYSignaler'])
		#utility = pd.DataFrame(index = self.modelNames )
		#utilityME = pd.DataFrame(index=self.modelNames)

		minItems = min(dfItems['nTargets'])
		maxItems = max(dfItems['nTargets'])+1

		for itemSetSize in range(minItems, maxItems):
			df_item = dfItems.loc[dfItems['nTargets'] == itemSetSize]
			print("items: ", itemSetSize, 'data points: ', df_item.shape[0])

			df_acc = self.getProportionTargetReached(df_item)
			accLabel = 'recAcc_' + str(itemSetSize)
			accuracy[accLabel] = df_acc['receiver']
			meLab = 'ME_'+ str(itemSetSize)
			accuracyME[meLab] = df_acc['marginOfErrorR']

			if self.utilPercent:
				df_util = self.getPercentFromOptimalUtilityDF(df_item)
			else:
				df_util = self.getDifferenceFromOptimalUtilityDF(df_item)
			utilSummary = self.getUtilityDifferenceSummaryStatistics(df_util)
			utilLabel = 'util' + str(itemSetSize)
			utility[utilLabel] = utilSummary['means']
			utilityME[meLab] = utilSummary['marginOfError']

		return(accuracy.transpose(), accuracyME.transpose(),utility.transpose(), utilityME.transpose())

	"""	
	def facetAccuracyByVocab(self, df):
		checkFullVocab = lambda x: self.checkFullVocab(x['signalSpace'], x['targetDictionary'])
		df['fullVocab'] = df.apply(checkFullVocab, axis=1)


	def checkFullVocab(self, signalSpace, targetDict):
		allFeatures = [tgt.split() for tgt in targetDict.values()]
		flatten = lambda l: [item for sublist in l for item in sublist]
		fullFeatureSignalSpace = list(set(flatten(allFeatures)))

		allSignalsPresent = all([sig in signalSpace for sig in fullFeatureSignalSpace])
		return(allSignalsPresent)
	"""










"""def getDifferenceFromMaximumUtilityDF(df):
    differenceInUtility = pd.DataFrame(df['CentralControl_utility']-df['IW_N_utility']).rename(columns = {0:'IW Naive'})
    differenceInUtility['IW Pragmatic'] = df['CentralControl_utility']-df['IW_P_utility']
    differenceInUtility['RSA Naive Language'] = df['CentralControl_utility']-df['RSA_NL_utility']
    differenceInUtility['RSA Pragmatic Language'] = df['CentralControl_utility']-df['RSA_PL_utility']
    differenceInUtility['RSA Pragmatic Action'] = df['CentralControl_utility']-df['RSA_PA_utility']
    differenceInUtility['Joint Utility'] = df['CentralControl_utility']-df['JU_utility']
    return(differenceInUtility)

def getUtilityDifferenceSummaryStatistics(df):
    summaryDF = pd.DataFrame(df.mean(axis=0)).rename(columns={0:'means'})
    summaryDF['stds'] = df.std(axis=0)
    numObs = df.shape[0]
    summaryDF['marginOfError'] = 1.96*summaryDF['stds']/np.sqrt(numObs)
    summaryDF['CI_low'] = summaryDF['means'] - summaryDF['marginOfError']
    summaryDF['CI_high'] = summaryDF['means'] + summaryDF['marginOfError']
    return(summaryDF)

def getProportionTargetReached(df):
    propTrialsTargetReached = pd.DataFrame(columns=['signaler', 'receiver'])
    getPropTrue =  lambda colName: df[colName].value_counts(normalize=True).loc[True] if (True in df[colName].value_counts(normalize=True).index) else 0.0

    propTrialsTargetReached.loc['IW Naive'] = [getPropTrue('IW_N_sAchievesGoal'), getPropTrue('IW_N_rAchievesGoal')]
    propTrialsTargetReached.loc['IW Pragmatic'] = [getPropTrue('IW_P_sAchievesGoal'), getPropTrue('IW_P_rAchievesGoal')]
    propTrialsTargetReached.loc['RSA Naive Language'] = [getPropTrue('RSA_NL_sAchievesGoal'), getPropTrue('RSA_NL_rAchievesGoal')]
    propTrialsTargetReached.loc['RSA Pragmatic Language'] = [getPropTrue('RSA_PL_sAchievesGoal'), getPropTrue('RSA_PL_rAchievesGoal')]
    propTrialsTargetReached.loc['RSA Pragmatic Action'] = [getPropTrue('RSA_PA_sAchievesGoal'), getPropTrue('RSA_PA_rAchievesGoal')]
    propTrialsTargetReached.loc['Joint Utility'] = [getPropTrue('JU_sAchievesGoal'), getPropTrue('JU_rAchievesGoal')]
    
    propTrialsTargetReached['total'] = propTrialsTargetReached['signaler'] + propTrialsTargetReached['receiver']

    propTrialsTargetReached['marginOfErrorS'] =  1.96*np.sqrt((propTrialsTargetReached['signaler']*(1-propTrialsTargetReached['signaler']))/df.shape[0])
    propTrialsTargetReached['marginOfErrorR'] =  1.96*np.sqrt((propTrialsTargetReached['receiver']*(1-propTrialsTargetReached['receiver']))/df.shape[0])

    return(propTrialsTargetReached)

def makeNItemModelComparison(df, minNItems = 2, maxNItems = 10):
    dfItems = df.copy()
    dlen = lambda x: len(x['targetDictionary'])
    dfItems['nTargets'] = dfItems.apply(dlen, axis=1) # gets the number of items in the target Dictionary

    newColumnNames = ['IW Naive', 'IW Pragmatic','RSA Naive Language','RSA Pragmatic Language','RSA Pragmatic Action','Joint Utility']
    accuracy = pd.DataFrame(index=newColumnNames)
    marginOfError = pd.DataFrame(index=newColumnNames)

    for i in range(minNItems,maxNItems):
        df_item = dfItems.loc[dfItems['nTargets'] == i]
        df_acc = getProportionTargetReached(df_item)

        colLab = 'recAcc_' + str(i)
        accuracy[colLab] = df_acc['receiver']

        meLab = 'ME_'+ str(i)
        marginOfError[meLab] = df_acc['marginOfErrorR']
    return(accuracy.transpose(), marginOfError.transpose())"""





def allTargetFeaturesInSignalSpace(signalSpace, targetDict):
    allFeatures = [tgt.split() for tgt in targetDict.values()]
    flatten = lambda l: [item for sublist in l for item in sublist]
    fullFeatureSignalSpace = list(set(flatten(allFeatures)))
    
    allSignalsPresent = all([sig in signalSpace for sig in fullFeatureSignalSpace])
    return(allSignalsPresent)


def plotAverageBars(summaryDF, save = False, filename = './utilityMeans.png', cols = ['#47802b'], fSize=(7,6)):
    fig = plt.figure(figsize=fSize)
    ax = fig.add_axes([0,0,1,1])
    modelNames = list(summaryDF.index)
    utilityDifferenceMeans = summaryDF['means'].values
    marginOfError = summaryDF['marginOfError'].values
    ax.bar(x = modelNames, height = utilityDifferenceMeans, width = 1, yerr = marginOfError, color = cols,edgecolor='white')
    ax.set_xticklabels(modelNames, rotation=90)
    if save:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plotAccuracyStackedBars(df, save=False, filename='./accuracies.png', rColor = '#484D60', sColor = '#9FA5BD', fSize = (7,6)):
    fig = plt.figure(figsize =fSize)
    ax = fig.add_axes([0,0,1,1])
    width=.5
    
    modelNames = list(df.index)
    y_signalerAchievesGoal = df['signaler'].values
    y_receiverAchievesGoal = df['receiver'].values
    me_rec = df['marginOfErrorR']
    me_sig = df['marginOfErrorS']
    
    
    p1 = plt.bar(modelNames, y_receiverAchievesGoal, width, yerr = me_rec, label = 'Receiver', color = rColor,edgecolor='white')
    p2 = plt.bar(modelNames, y_signalerAchievesGoal, width,yerr = me_sig, label = 'Signaler', color = sColor, edgecolor='white',bottom=y_receiverAchievesGoal)
    
    ax.set_yticks([0,.5,1])
    ax.set_xticklabels(modelNames, rotation=90)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    if save:
        fig.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()


def plotReceiverSuccessfulCommunication(df, save=False, filename='./propSuccessfulCom.png', rColor = '#484D60', fSize = (7,6)):
    fig = plt.figure(figsize =fSize)
    ax = fig.add_axes([0,0,1,1])
    width=.5
    
    modelNames = list(df.index)
    y_receiverAchievesGoal = df['receiver'].values
    me_rec = df['marginOfErrorR']
    p1 = plt.bar(modelNames, y_receiverAchievesGoal, width,yerr=me_rec,label = 'Receiver', color = rColor,edgecolor='white')
    
    ax.set_yticks([0,.5,1])
    ax.set_xticklabels(modelNames, rotation=90)
    
    if save:
        fig.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()



def plotMetricByItem(accDf, meDF, save = False, filename = './acc.png', yAxisLabel = 'Proportion successful communication', yTicks = [0,.1, .2, .3, .4, .5,.6, .7, .8, .9, 1], modelNames=['IW Pragmatic', 'RSA Pragmatic Action'], fSize= (8,5),minItems = 2, maxItems = 9):
	fig = plt.figure(figsize =fSize)
	ax = fig.add_axes([0,0,1,1])
	labelDict = {'IW Naive':'IW Naive', 
				'IW Pragmatic':'IW',
				'RSA Naive Language':'RSA Naive Language',
				'RSA Pragmatic Language':'RSA Pragmatic Language',
				'RSA Pragmatic Action':'RSA',
				'Joint Utility':'Joint Utility', 
				"DIYSignaler": 'Signaler Does', 
				'Optimal': 'Optimal',
				'R0 IW Naive':'R0 IW Naive',
				'R0 IW Pragmatic': 'R0 IW Pragmatic',
				'R0 RSA Naive Language': 'R0 RSA Naive Language',
				'R0 RSA Pragmatic Language': 'R0 RSA Pragmatic Language',
				'R0 RSA Pragmatic Action': 'R0 RSA Pragmatic Action',
				'R0 Joint Utility':'R0 Joint Utility'}
				
	colorDict = {'IW Naive':'#b10d2f', 
				'IW Pragmatic':'#b10d2f', 
				'RSA Naive Language':'#fed554', 
				'RSA Pragmatic Language': '#fed554', 
				'RSA Pragmatic Action':'#fed554',
				'Joint Utility':'#95a84c', 
				"DIYSignaler": '#000650', 
				'Optimal': '#555555'} 

	modelLabels = accDF.columns if modelNames == 'all' else modelNames

	x = [a for a in range(minItems, maxItems+1)]

	for modelLabel in modelLabels:
		plt.errorbar(x, accDf[modelLabel].values, yerr = meDF[modelLabel].values, color = colorDict[modelLabel],label=labelDict[modelLabel], linewidth=3)
	plt.xlabel('Number of Items')
	plt.ylabel(yAxisLabel)
	plt.legend(loc='best')
	ax.set_yticks(yTicks)
	ax.tick_params(labelsize=24)
	if save:
		fig.savefig(filename,dpi=300, bbox_inches='tight')
	plt.show()