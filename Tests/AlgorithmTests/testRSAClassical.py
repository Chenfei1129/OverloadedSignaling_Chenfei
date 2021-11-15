import sys
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import unittest
from ddt import ddt, data, unpack

import Algorithms.RSA.RSAClassical as targetCode

@ddt
class TestRSALexicon(unittest.TestCase): 
        def setUp(self):
                pass

        @data(('triangle', 'green triangle'), ('triangle', 'purple triangle'), ('green', 'green triangle'), ('green', 'green circle'))
        @unpack
        def test_RSALexicon_consistentMessageIsPartialTarget(self, message, target):
                messageIsConsistent  = targetCode.consistencyLexiconRSA_Experiment(message, target)
                self.assertTrue(messageIsConsistent)

        @data(('green triangle', 'green triangle'), ('purple circle', 'purple circle'), ('triangle green', 'green triangle'))
        @unpack
        def test_RSALexicon_consistentMessageIsTarget(self, message, target):
                messageIsConsistent  = targetCode.consistencyLexiconRSA_Experiment(message, target)
                self.assertTrue(messageIsConsistent)

        @data(('triangle', 'green circle'), ('triangle', 'purple circle'), ('green', 'purple triangle'))
        @unpack
        def test_RSALexicon_inconsistentMessage(self, message, target):
                messageIsConsistent  = targetCode.consistencyLexiconRSA_Experiment(message, target)
                self.assertFalse(messageIsConsistent)

        def tearDown(self):
                pass

@ddt
class TestRSAModeling_ListenerEntryPoint(unittest.TestCase):
        def setUp(self): 
                self.lexicon = targetCode.consistencyLexiconRSA_Experiment

                self.uniformPrior = {'green triangle': 1.0/3, 'green circle':1.0/3 , 'purple circle':1.0/3}
                self.getLiteralListener = targetCode.LiteralListener(self.uniformPrior, self.lexicon)
                self.getCost = targetCode.costOfMessage

                oneOverloadedMessage = ['purple', 'circle']
                twoOverloadedMessages = ['green', 'circle']
                fullVocab = ['purple', 'green', 'circle', 'triangle']
                rationality = 1

                self.getPragmaticSpeaker_OneOverloaded = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, 
                                                                                                                                                messageSet=oneOverloadedMessage, 
                                                                                                                                                messageCostFunction=self.getCost, 
                                                                                                                                                lambdaRationalityParameter=rationality)

                self.getPragmaticSpeaker_TwoOverloaded = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, 
                                                                                                                                                messageSet=twoOverloadedMessages, 
                                                                                                                                                messageCostFunction=self.getCost, 
                                                                                                                                                lambdaRationalityParameter=rationality)

                self.getPragmaticSpeaker_FullVocab = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, 
                                                                                                                                                messageSet=fullVocab, 
                                                                                                                                                messageCostFunction=self.getCost, 
                                                                                                                                                lambdaRationalityParameter=rationality)

                
        #########################################################################
        ######## LITERAL SPEAKER
        #########################################################################
                
        @data(('green triangle', 'green', 1.0), ('purple circle', 'circle', 1.0), ('green circle', 'green', .5), ('green circle',  'circle', .5))
        @unpack
        def test_literalSpeaker_consistentItems_TwoOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['green', 'circle']
                rationality = 1
                getLiteralSpeaker = targetCode.LiteralSpeaker(lexicon=self.lexicon, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)
                literalSpeaker = getLiteralSpeaker(target)
                probabilityOfMessage = literalSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)
                
        @data(('green triangle', 'circle', 0.0), ('purple circle', 'green', 0.0))
        @unpack
        def test_literalSpeaker_inconsistentItems_TwoOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['green', 'circle']
                rationality = 1
                getLiteralSpeaker = targetCode.LiteralSpeaker(lexicon=self.lexicon, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)
                literalSpeaker = getLiteralSpeaker(target)
                probabilityOfMessage = literalSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)
        
        @data(('green circle', 'circle', 1.0), ('purple circle', 'purple', 1.0/2), ('purple circle', 'circle', 1.0/2))
        @unpack
        def test_literalSpeaker_consistentItems_OneOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['purple', 'circle']
                rationality = 1
                getLiteralSpeaker = targetCode.LiteralSpeaker(lexicon=self.lexicon, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)
                literalSpeaker = getLiteralSpeaker(target)
                probabilityOfMessage = literalSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)

        @data(('green circle', 'purple', 0.0))
        @unpack
        def test_literalSpeaker_inconsistentItems_OneOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['purple', 'circle']
                rationality = 1
                getLiteralSpeaker = targetCode.LiteralSpeaker(lexicon=self.lexicon, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)
                literalSpeaker = getLiteralSpeaker(target)
                probabilityOfMessage = literalSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)
                
        @data(('green triangle', 'circle', 0.5), ('green triangle', 'purple', 0.5))
        @unpack
        def test_literalSpeaker_inconsistentItems_OneOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['purple', 'circle']
                rationality = 1
                getLiteralSpeaker = targetCode.LiteralSpeaker(lexicon=self.lexicon, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)
                literalSpeaker = getLiteralSpeaker(target)
                probabilityOfMessage = literalSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)
                
                
        #########################################################################
        ######## LITERAL LISTENER
        #########################################################################
        @data(('green', 'green triangle', .5), ('purple', 'green triangle', 0), ('circle', 'green circle', .5))
        @unpack
        def test_literalListener_uniformPrior_Experiment(self, message, targetToTest, expectedResult):
                uniformPrior = {'green triangle': 1.0/3, 'green circle':1.0/3 , 'purple circle':1.0/3}
                getLiteralListener = targetCode.LiteralListener(uniformPrior, self.lexicon)

                literalListener = getLiteralListener(message = message)
                probabilityOfTarget= literalListener.loc[targetToTest].values[0]
                self.assertAlmostEqual(probabilityOfTarget, expectedResult)

        @data(('green', 'green triangle', .5), ('purple', 'green triangle', 0), ('circle', 'green circle', .5))
        @unpack
        def test_literalListener_fourItemPrior_Experiment(self, message, targetToTest, expectedResult):
                fourItemPrior = {'green triangle': 1.0/4, 'green circle':1.0/4 , 'purple circle':1.0/4, 'purple triangle': 1.0/4}
                getLiteralListener = targetCode.LiteralListener(fourItemPrior, self.lexicon)

                literalListener = getLiteralListener(message = message)
                probabilityOfTarget= literalListener.loc[targetToTest].values[0]
                self.assertAlmostEqual(probabilityOfTarget, expectedResult)

        @data(('green', 'green triangle', .5), ('purple', 'green triangle', 0), ('circle', 'green circle', 1.0/3), ('circle', 'purple circle', 2.0/3))
        @unpack
        def test_literalListener_nonUniformPrior_Experiment(self, message, targetToTest, expectedResult):
                nonUniformPrior = {'green triangle': 1.0/4, 'green circle':1.0/4 , 'purple circle':1.0/2}
                getLiteralListener = targetCode.LiteralListener(nonUniformPrior, self.lexicon)

                literalListener = getLiteralListener(message = message)
                probabilityOfTarget= literalListener.loc[targetToTest].values[0]
                self.assertAlmostEqual(probabilityOfTarget, expectedResult)

        @data(('glasses', 'glasses', .5), ('hat', 'glasses', 0), ('glasses', 'glasses hat', .5))
        @unpack
        def test_literalListener_uniformPrior_RSAClassicExample(self, message, targetToTest, expectedResult):
                uniformPrior = {' ': 1.0/3, 'glasses':1.0/3 , 'glasses hat':1.0/3}
                getLiteralListener = targetCode.LiteralListener(uniformPrior, self.lexicon)

                literalListener = getLiteralListener(message = message)
                probabilityOfTarget= literalListener.loc[targetToTest].values[0]
                self.assertAlmostEqual(probabilityOfTarget, expectedResult)

        
        #########################################################################
        ######## PRAGMATIC SPEAKER
        #########################################################################
        @data(('green triangle', 'green', 1.0), ('purple circle', 'circle', 1.0), ('green circle', 'green', .5), ('green circle',  'circle', .5))
        @unpack
        def test_pragmaticSpeaker_consistentItems_TwoOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['green', 'circle']
                rationality = 1
                getPragmaticSpeaker = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)
                pragSpeaker = getPragmaticSpeaker(target)
                probabilityOfMessage= pragSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)

        @data(('green triangle', 'circle', 0.0), ('purple circle', 'green', 0.0))
        @unpack
        def test_pragmaticSpeaker_inconsistentItems_TwoOverloadedSignals_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['green', 'circle']
                rationality = 1
                getPragmaticSpeaker = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)

                pragSpeaker = getPragmaticSpeaker(target)
                probabilityOfMessage= pragSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)

        

        @data(('green circle', 'circle', 1.0), ('purple circle', 'purple', 2.0/3), ('purple circle', 'circle', 1.0/3))
        @unpack
        def test_pragmaticSpeaker_consistentItems_OneOverloadedSignal_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['purple', 'circle']
                rationality = 1
                getPragmaticSpeaker = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)

                pragSpeaker = getPragmaticSpeaker(target)
                probabilityOfMessage= pragSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)

        @data(('green circle', 'purple', 0.0))
        @unpack
        def test_pragmaticSpeaker_inconsistentItems_OneOverloadedSignal_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['purple', 'circle']
                rationality = 1
                getPragmaticSpeaker = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality)

                pragSpeaker = getPragmaticSpeaker(target)
                probabilityOfMessage= pragSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)


        #Original formulation for RSA makes it so that if there are NO consistent signals for a possible target, the signaler chooses uniformly among the options
        @data(('green triangle', 'circle', 0.5), ('green triangle', 'purple', 0.5))
        @unpack
        def test_pragmaticSpeaker_noPossibleConsistentSignal_OneOverloadedSignal_Experiment(self, target, messageToTest, expectedResult):
                omegaMessages = ['purple', 'circle']
                rationality = 1
                getPragmaticSpeaker = targetCode.PragmaticSpeaker(getListener=self.getLiteralListener, messageSet=omegaMessages, messageCostFunction=self.getCost, lambdaRationalityParameter=rationality) 
                pragSpeaker = getPragmaticSpeaker(target)
                probabilityOfMessage= pragSpeaker.loc[messageToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)


        #########################################################################
        ######## PRAGMATIC LISTENER
        #########################################################################
        @data(('green', 'green circle', .6), ('purple', 'purple circle', 1.0), ('circle', 'purple circle', .4))
        @unpack
        def test_pragmaticListener_fullVocab_Experiment(self, message, targetToTest, expectedResult):

                getPragmaticListener = targetCode.PragmaticListener(getSpeaker=self.getPragmaticSpeaker_FullVocab, targetPrior=self.uniformPrior)

                pragSpeaker = getPragmaticListener(message)
                probabilityOfMessage = pragSpeaker.loc[targetToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)


        @data(('green', 'green circle', 1.0/3), ('circle', 'purple circle', 2.0/3))
        @unpack
        def test_pragmaticListener_twoOverloadedSignals_Experiment(self, message, targetToTest, expectedResult):

                getPragmaticListener = targetCode.PragmaticListener(getSpeaker=self.getPragmaticSpeaker_TwoOverloaded, targetPrior=self.uniformPrior)

                pragSpeaker = getPragmaticListener(message)
                probabilityOfMessage = pragSpeaker.loc[targetToTest].values[0]
                self.assertAlmostEqual(probabilityOfMessage, expectedResult)


        def tearDown(self):
                pass



if __name__ == '__main__':
        unittest.main(verbosity=2)
