import os
import sys      
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..','..'))

import numpy as np
import pandas as pd
import itertools
import warnings

import Algorithms.constantNames as NC
import Simulations.modelLabels_5_dimensions as ML

from Simulations.setupModelInferences_moreWord import SetupModels
from Simulations.setupSampledTrials_MultipleSignals import SampleExperimentEnvironment4
from Simulations.setupMetrics2 import SimulateModelFromDF, SingleAgentUtility_TaxicabMetric
from Environments.Experiment.experimentConstruction import calculateLocationCost_TaxicabMetric
from multiprocessing import Pool
from functools import partial
from Simulations.parallelComputing import parallelize_dataframe, run_on_subset, parallelize_on_row 
def main():
        randomSeed = 2822
        filename = './simulation.pkl'
        modNames = ['IW_S1R0' ]#,'IW_S1R0' # 'RSA_S1R1','RSA_S1R0','RSA_S0R0'
        #sim_r0 = pd.read_pickle('./simulation_signalerCanQuit_a4_r8_seed282_variedItemFullVocab_Chenfei_three_features_Comparison_More_Added_maxSampling_4_dimensions_random_items_new_one_cost_new_feature_more.pkl')
        #Fixed Parameters
        alpha = 10
        rewardValue = 8
        nruns = 60
        cores = 6

        #Sampling spaces
        gridDims = (9,10) #0-8, by 0-9
        s = (4,0)
        r = (4,7)
        fixedEnvironmentParameters = {'gridSize': gridDims, 'signalerPosition': s, 'receiverPosition': r}
        
        itemSpace = ML.itemSpace_3Color3Shape
        signalSpace = ML.featureSpace_3Color3Shape
        fluidParameterSpaces = {'targets': itemSpace,'signals': signalSpace}

        #Parameters specific to this simualation:
        signalProp = 1.0
        prop = True

        # Trial Setup
        np.random.seed(randomSeed)
        getEnvironment = SampleExperimentEnvironment4(fixedEnvironmentParameters, fluidParameterSpaces)
        simulatedTrials = pd.DataFrame(columns = ML.SAMPLED_TRIAL_PARAMETERS)

        for indx in range(nruns):
                #goal, sigSpace, targetD, nTargets, relevantSignalProp = getEnvironment(numItem = nItems, numberOfSignals = signalProp, vocabProportion = prop)
                #goal = sim_r0['intention'][indx]
                #sigSpace = sim_r0['signalSpace'][indx]
                #targetD = sim_r0['targetDictionary'][indx]
                #nTargets = sim_r0['nTargets'][indx]
                #relevantSignalProp = sim_r0['propRelevantSignalsInVocab'][indx]
                #simulatedTrials.loc[indx] = [s, r, goal, sigSpace, targetD, nTargets, relevantSignalProp]
                nItems = np.random.choice(range(3,7)) * 2
                goal, sigSpace, targetD, nTargets, relevantSignalProp = getEnvironment(numItem = nItems, numberOfSignals = signalProp, vocabProportion = prop)
                simulatedTrials.loc[indx] = [s, r, goal, sigSpace, targetD, nTargets, relevantSignalProp]

#Pass old environment change to 0.1
        # Modeling inference setup
        params_a4r8 = {'rationality' : alpha, 
                                        'reward': rewardValue, 
                                        'costFunction' : calculateLocationCost_TaxicabMetric,
                                        'signalerInaction': True}
        fullSimulations = simulatedTrials.copy()
        k = simulatedTrials.copy()
        
        signalCost = [0, 0.1]#, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2
        for cost in signalCost:
            #signalerRatio = 1-ratio
            getSAUtil = SingleAgentUtility_TaxicabMetric(reward = rewardValue)
        
            params_a4r8 = {'rationality':alpha, 'reward': rewardValue, 'costFunction' : calculateLocationCost_TaxicabMetric, 'signalCost': cost, 'signalerInaction':True}
            setupModInference = SetupModels(modelParameters=params_a4r8, modelNames = modNames )
            #getReceiverUtil = SingleAgentUtility_TaxicabMetric_Ratio(reward = rewardValue, ratioCost = ratio)

            getModelMetrics = SimulateModelFromDF(buildModels=setupModInference, getUtility= getSAUtil, signalerCanQuit = True )
            costSignalReceiverSimulation = parallelize_on_row(fullSimulations, getModelMetrics, cores)
            #costSignalReceiverSimulation = fullSimulations.apply(getModelMetrics, axis = 1)
            suffixLabel = 'S_C' + str(cost) 
        
            k = k.join(costSignalReceiverSimulation, rsuffix=suffixLabel)
        

        #Save full simulated data
        k.to_pickle('./simulation_5_dimensions_5_features_central_controller_round_6.pkl')

if __name__ == "__main__":
        main()
