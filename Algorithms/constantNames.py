#names of components of the mind
WORLDS = 'worlds'
ACTIONS = 'actions'
DESIRES = 'desires'
INTENTIONS = 'intentions'
SIGNALS = 'signals'

#utility based
UTILITY = 'utility'
PROBABILITY = 'probability'

# Extension for signaler Type
SINGALER_C = 'signalerType' 

#names of mind component probabilities
P_WORLD = 'p(w)'
P_DESIRE = 'p(d)'
P_INTENTION =  'p(i|w,d)'
P_ACTION = 'p(a|w,i)'

#joint and marginal probabilities
P_MIND = 'p(mind)'
P_C = 'p(c)'
P_JOINTPRIOR = 'p(mind,c)'
P_SIGNALLIKELIHOOD = 'p(signal|mind,c)'
P_SIGNALONLYLIKELIHOOD = 'p(signal|mind)'
P_MINDPOSTERIOR = 'p(mind|signal)'


# RSA Specific
LITERALLISTENER = 'l0(w|msg)'
LITERALSPEAKER = 's0(msg|w)'
PRAGMATICLISTENER = 'l1(w|msg)'
PRAGMATICSPEAKER = 's1(msg|w)'
