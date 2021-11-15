import os
import sys      
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import numpy as np
import pandas as pd
import Simulations.modelLabels as ML
from Simulations.setupModelInferences_CostRatioReceiver import SetupModels_CostRatioReceiver
from Simulations.setupModelInferences import SetupModels
from Simulations.setupSampledTrials import SampleExperimentEnvironment
from Simulations.setupMetrics import SimulateModelFromDF, SingleAgentUtility_TaxicabMetric, SingleAgentUtility_TaxicabMetric_Ratio, CostlessReceiverUtility#
from multiprocessing import Pool
from functools import partial
from Simulations.parallelComputing import parallelize_dataframe, run_on_subset, parallelize_on_row 
def main():
        randomSeed = 423
        filename = './simulation_costRaioReceiver_a4_r8_seed423_2-9ItemFullVocab_2000_JU_report__signaler.pkl'
        modNames = ['JU' ]
        
        #Fixed Parameters
        alpha = 4
        rewardValue = 4
        cores = 4
        #params_a4r8 = {'rationality':alpha, 'reward': rewardValue, 'ratio': ratio}
        nruns = 2000

        #Sampling spaces
        gridDims = (9,10) #0-8, by 0-9
        s = (4,0)
        r = (4,7)
        fixedEnvironmentParameters = {'gridSize': gridDims, 'signalerPosition': s, 'receiverPosition': r}
        
        itemSpace = ML.itemSpace_3Color3Shape
        signalSpace = ML.featureSpace_3Color3Shape
        #receiverCost = [0, 0.5, 1]
        fluidParameterSpaces = {'targets': itemSpace,'signals': signalSpace}#, 'receiverCostRatio':receiverCost
        #Parameters specific to this simualation:
        signalProp = 1.0
        prop = True

        # Trial Setup
        np.random.seed(randomSeed)
        getEnvironment = SampleExperimentEnvironment(fixedEnvironmentParameters, fluidParameterSpaces)
        simulatedTrials = pd.DataFrame(columns = ML.SAMPLED_TRIAL_PARAMETERS)

        for indx in range(nruns):
                nItems = np.random.choice(range(2,10))#9,10
                goal, sigSpace, targetD, nTargets, relevantSignalProp = getEnvironment(numberOfPossibleTargets = nItems, numberOfSignals = signalProp, vocabProportion = prop)
                simulatedTrials.loc[indx] = [s, r, goal, sigSpace, targetD, nTargets, relevantSignalProp]

        #Cost Receiver
        #setupModInference = SetupModels(modelParameters=params_a4r8, modelNames = modNames)
        

        #getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility= getSAUtil)
        #rsaReceiverSimulation = simulatedTrials.apply(getModelMetrics, axis=1)
        #costReceiverWithTrialInfo = simulatedTrials.join(rsaReceiverSimulation)
        fullSimulations = simulatedTrials.copy()
        k = simulatedTrials.copy() 

        #CostRatio Receiver
        costRatio = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
        for ratio in costRatio:
            signalerRatio = 1-ratio
            getSAUtil = SingleAgentUtility_TaxicabMetric_Ratio(reward = rewardValue, ratioCost = signalerRatio)
        
            params_a4r8 = {'rationality':alpha, 'reward': rewardValue, 'receiverCostRatio': ratio, 'signalerInaction':True}
            setupModInference_C0R = SetupModels_CostRatioReceiver(modelParameters=params_a4r8, modelNames = modNames)
            getReceiverUtil = SingleAgentUtility_TaxicabMetric_Ratio(reward = rewardValue, ratioCost = ratio)

            getModelMetrics_R0 = SimulateModelFromDF(buildModels=setupModInference_C0R, getUtility= getSAUtil, receiverSpecificUtility = getReceiverUtil)
            costRatioReceiverSimulation = parallelize_on_row(fullSimulations, getModelMetrics_R0, cores)
            #costRatioReceiverSimulation = fullSimulations.apply(getModelMetrics_R0, axis = 1)
            suffixLabel = 'C_R' + str(ratio) 
        

            k = k.join(costRatioReceiverSimulation, rsuffix=suffixLabel)
        #fullSimulations = costReceiverWithTrialInfo.join(costlessReceiverSimulation, rsuffix='_C0R')

        #Save full simulated data
        k.to_pickle(filename)

if __name__ == "__main__":
        main()
