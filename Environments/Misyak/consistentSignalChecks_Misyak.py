import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import Algorithms.constantNames as NC

# Misyak - returns a boolean of whether the input signal is possible given the mind and signal category
def signalIsConsistent_Boxes(signal, mind, signalMeaning, openSignalMeaning = '1'):
    world = mind[NC.WORLDS]
    signalTypeStr = str(signalMeaning)
    
    if signalTypeStr == openSignalMeaning:
        consistentSignals = [int(w == 1) for token, w in zip(signal, world) if token == 1]  
    else:
        consistentSignals = [int(w != 1) for u, w in zip(signal, world) if u == 1]
    return(consistentSignals.count(0) == 0)



