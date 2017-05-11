import unittest
import TFlibs.networkcrafter as nc
import tensorflow as tf
import numpy as np

def testDropout(layer, expectDropoutOn):
   if expectDropoutOn:
        return layer.applyDropout and (type(layer.keepProb) is tf.Tensor)
   else:
        return (not layer.applyDropout)

class TestLayerInitialization(unittest.TestCase):

    def test_base_layer_init(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            # Layer with no dropout, no activation
            shape = [10, 4]
            layer = nc.Layer(shape)
            self.assertEqual(shape, layer.shape)
            self.assertFalse(layer.applyDropout)
            self.assertEqual(None, layer.activationFunction)

            # Layer with dropout and activation
            shape = [4, 10]
            layer = nc.Layer(shape, tf.nn.softmax, True)
            self.assertEqual(shape, layer.shape)
            self.assertTrue(layer.applyDropout)
            self.assertEqual(layer.activationFunction, tf.nn.softmax)

    def test_input_layer_init(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            nFeatures = 10
            setShape = [2, nFeatures]
            layer = nc.InputLayer(nFeatures, applyOneHot=False, dtype=tf.int32)
            feed = {layer.inputs:np.ones(shape=setShape)}

            self.assertTrue(layer.shape == [None, nFeatures])
            self.assertTrue(testDropout(layer, False))
            shape = tf.shape(layer.activations).eval(session=sess, feed_dict=feed)
            self.assertTrue(shape.all() == np.array(setShape).all())


class TestLayerGraphBuild(unittest.TestCase):

    def test_base_layer_build(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            shape = [2, 4]
            # Layer with no dropout, with activation
            layer = nc.Layer(shape, tf.nn.softmax)
            with self.assertRaises(TypeError):
                layer.buildGraph(shape)
            layer.buildGraph(tf.constant(np.ones(shape)))
            self.assertTrue(isinstance(layer.activations, tf.Tensor))
            manuallyActivated = sess.run(tf.nn.softmax(layer.weightedInputs))
            self.assertEqual(sess.run(layer.activations).tolist(), manuallyActivated.tolist())

            shape = [7, 8]
            # Layer with no dropout, no activation
            layer = nc.Layer(shape)
            layer.buildGraph(tf.constant(np.ones(shape)*5, dtype=tf.float32))
            self.assertEqual(sess.run(layer.activations).tolist(), (np.ones(shape)*5).tolist())

            shape = [100, 100]
            # Layer with dropout, no activation
            layer = nc.Layer(shape, dropout=True)
            layer.buildGraph(tf.ones(shape=shape))
            self.assertTrue(isinstance(layer.keepProb, tf.Tensor))
            outputs = sess.run(layer.activations, feed_dict={layer.keepProb:1})
            self.assertEqual(np.sum(outputs), shape[0]*shape[1])
            outputs = sess.run(layer.activations, feed_dict={layer.keepProb:0.5})
            numKept = np.sum(np.where(outputs == 0, 0, 1))
            numExpected =  (shape[0]*shape[1])*0.5
            self.assertTrue(abs(numKept-numExpected) < 100)


class TestLayerOutputs(unittest.TestCase):

    def test_inputLayer_basic(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            nIn = 9

            net = nc.Network()
            net.inputLayer(nIn)
            net.buildGraph()
            inLayer = net.layers[0]

            inputs = np.array([[1,2,3,4,5,6,7,8,9]]*13)
            outputs = sess.run(net.outputs, feed_dict={inLayer.inputs:inputs})

            self.assertTrue(inputs.tolist() == outputs.tolist())

    def test_inputLayer_oneHot(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            nIn = 7

            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True)
            net.buildGraph()
            inLayer = net.layers[0]

            # -1, and 7 are "out of bounds" for one-hot with 7 classes. Should get zero-vectors.
            inputs = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7])
            zeros = np.zeros([1,7])
            expectedOut = np.concatenate([zeros, np.eye(7), zeros], axis=0)

            outputs = sess.run(net.outputs, feed_dict={inLayer.inputs:inputs})
            self.assertTrue(outputs.tolist() == expectedOut.tolist())

    def test_inputLayer_tensorIn(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            nIn = 5

            inputs = np.array([[1,2,3,4,5]]*22)

            net = nc.Network()
            net.inputLayer(nIn, inputTensor=tf.identity(inputs))
            net.buildGraph()
            inLayer = net.layers[0]

            outputs = sess.run(net.outputs)

            self.assertTrue(inputs.tolist() == outputs.tolist())

    def test_inputLayer_tensorIn_oneHot(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            nIn = 3

            inputs = np.array([-1, 0, 1, 2, 3])

            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True, inputTensor=tf.identity(inputs))
            net.buildGraph()
            inLayer = net.layers[0]

            # -1 and 3 are "out of bounds" for one-hot with 3 classes. Should get zero-vectors.
            zeros = np.zeros([1,3])
            expectedOut = np.concatenate([zeros, np.eye(3), zeros], axis=0)

            outputs = sess.run(net.outputs)
            self.assertTrue(expectedOut.tolist() == outputs.tolist())


    def test_embeddingLayer(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            numEmbeddings = 100
            embedDim = 12

            net = nc.Network()
            net.embeddingLayer(numEmbeddings, embedDim)
            net.buildGraph()
            embedLayer = net.inLayer

            sess.run(tf.global_variables_initializer())

            # Some random embedding indices to ask for.
            ids = np.random.randint(0, numEmbeddings, 33)
            expectedOut = sess.run(embedLayer.embeddings)[ids, :]
            out = net.forward(sess, ids)
            self.assertTrue(expectedOut.tolist() == out.tolist())

    def test_embeddingLayer_tensorIn(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            numEmbeddings = 45
            embedDim = 12

            lookupTensor = tf.constant(np.random.randint(0, numEmbeddings, 22))

            netPlaceholder = nc.Network()
            netPlaceholder.embeddingLayer(numEmbeddings, embedDim)
            netPlaceholder.buildGraph()

            netExternalLookup = nc.Network(reuseVariables=True)
            netExternalLookup.embeddingLayer(numEmbeddings, embedDim, lookupTensor)
            netExternalLookup.buildGraph()
            embedLayer = netExternalLookup.outLayer

            sess.run(tf.global_variables_initializer())

            # Manually apply the lookup tensor and check that we get the same from the network
            expectedOut = sess.run(embedLayer.embeddings)[lookupTensor.eval()]
            out = netExternalLookup.forward(sess, None)
            self.assertTrue(expectedOut.tolist() == out.tolist())

            # Check that the two networks yield the same result.
            outFromPlaceholder = netPlaceholder.forward(sess, lookupTensor.eval())
            outFromExternal = netExternalLookup.forward(sess, None)
            self.assertEqual(outFromPlaceholder.tolist(), outFromExternal.tolist())


    def test_reduceOp(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            nIn = 5
            nExamples = 12

            inputs = np.random.rand(nIn, nExamples)

            meanNet = nc.Network("mean")
            meanNet.inputLayer(nIn, inputTensor=tf.constant(inputs))
            meanNet.reduceOpLayer(axis=1)
            meanNet.buildGraph()

            out = meanNet.outputs.eval()
            expectedOut = tf.reduce_mean(inputs, axis=1).eval()
            self.assertEquals(out.tolist(), expectedOut.tolist())
            self.assertEquals(out.shape[-1], nIn)

            sumNet = nc.Network("sum")
            sumNet.inputLayer(nIn, inputTensor=tf.constant(inputs))
            sumNet.reduceOpLayer(reduceOp=tf.reduce_sum)
            sumNet.buildGraph()

            out = sumNet.outputs.eval()
            expectedOut = tf.reduce_sum(inputs, axis=0).eval()
            self.assertEquals(out.tolist(), expectedOut.tolist())
            self.assertEquals(out.shape[-1], nExamples)

    def test_fullConnectLayer(self):

        # Basic Full Connect
        tf.reset_default_graph()
        with tf.Session() as sess:
            nNodes = 100
            nIn = 12
            inputs = range(nIn)
            hotInputs = np.eye(nIn)[inputs]

            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True)
            net.fullConnectLayer(nNodes, None)
            net.buildGraph()
            sess.run(tf.global_variables_initializer())

            fullLayer = net.layers[1]
            weights = fullLayer.weights.eval(sess)
            biases = fullLayer.biases.eval(sess)

            # Output should be as expected:  xW + b
            expectedOut = np.dot(hotInputs, weights) + biases
            out = net.forward(sess, inputs)

            # We allow negligible differences between numpy and tensorflow result
            maxDiff = 1e-08*out.shape[0]*out.shape[1]
            difference = np.sum(out - expectedOut)

            self.assertTrue(maxDiff >= difference)

        # No Bias
        tf.reset_default_graph()
        with tf.Session() as sess:
            nNodes = 7
            nIn = 46
            inputs = range(nIn)
            hotInputs = np.eye(nIn)[inputs]

            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True)
            net.fullConnectLayer(nNodes, None, addBias=False)
            net.buildGraph()
            sess.run(tf.global_variables_initializer())

            fullLayer = net.layers[1]
            weights = fullLayer.weights.eval(sess)

            # Output should be as expected:  xW
            expectedOut = np.dot(hotInputs, weights)
            out = net.forward(sess, inputs)

            # We allow negligible differences between numpy and tensorflow result
            maxDiff = 1e-08*out.shape[0]*out.shape[1]
            difference = np.sum(out - expectedOut)

            self.assertTrue(maxDiff >= difference)

        # Weight Transpose
        tf.reset_default_graph()
        with tf.Session() as sess:
            nNodes = 31
            nIn = 56
            inputs = range(nIn)
            hotInputs = np.eye(nIn)[inputs]

            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True)
            net.fullConnectLayer(nNodes, None, wTranspose=True)
            net.buildGraph()
            sess.run(tf.global_variables_initializer())

            fullLayer = net.layers[1]
            weights = fullLayer.weights.eval(sess)
            biases = fullLayer.biases.eval(sess)

            # Output should be as expected:  xW' + b
            expectedOut = np.dot(hotInputs, np.transpose(weights)) + biases
            out = net.forward(sess, inputs)

            # We allow negligible differences between numpy and tensorflow result
            maxDiff = 1e-08*out.shape[0]*out.shape[1]
            difference = np.sum(out - expectedOut)

            self.assertTrue(maxDiff >= difference)

    def test_GRUBasic_seqLenTensor(self):
        nNodes = 7
        maxSeqLen = 10
        batchSize = 3

        def getNetwork(sequenceLengths, reuse):
            net = nc.Network(reuseVariables=reuse)
            net.inputLayer(nNodes)
            net.basicGRU(nNodes=nNodes, batchSize=batchSize, maxSeqLen=maxSeqLen,
                sequenceLengths=sequenceLengths, saveState=False)
            net.buildGraph()
            return net

        tf.reset_default_graph()
        with tf.Session() as sess:
            seqLens = [2, 10, 4]
            netSeqLenIn = getNetwork(seqLens, False)

            netSeqLenPlace = getNetwork(None, True)

            sess.run(tf.global_variables_initializer())

            ins = np.random.rand(batchSize*maxSeqLen, nNodes)
            outSeqLenIn = netSeqLenIn.forward(sess, ins)
            outSeqLenPlace = netSeqLenPlace.forward(sess, ins, sequenceLengths=seqLens)
            # Same result when seq lens are given as placeholder or defined on layer init
            self.assertEquals(outSeqLenIn.tolist(), outSeqLenPlace.tolist())

    def test_GRUBasic_stateOut(self):

        nNodes = 2
        nIn = 4
        maxInLen = 10
        inputs = [1]*maxInLen
        inLength = 5

        def getNetwork(nLayers, scopeName):
            net = nc.Network(scopeName)
            net.inputLayer(nIn, applyOneHot=True)
            net.basicGRU(nNodes, nLayers=nLayers,  maxSeqLen=maxInLen, saveState=False,
                activationsAreFinalState=True)
            net.buildGraph()
            gru = net.outLayer
            return net, gru

        def doTests():

            sess.run(tf.global_variables_initializer())

            feed = net.getFeedDict(inputs, sequenceLengths=inLength)
            finalState, outSequence = sess.run([net.outputs, gru.outSequence[0]], feed)

            # Timestep reshaped to dim-0
            outSequence = np.reshape(outSequence, [-1, nNodes])
            lastOut = outSequence[inLength-1]

            # network output is the final state which is the output at time inLength
            self.assertTrue(lastOut.tolist() == finalState[0].tolist())

        tf.reset_default_graph()
        with tf.Session() as sess:

            net, gru = getNetwork(1, "singleLayer")
            doTests()

            net, gru = getNetwork(2, "multiLayer")
            doTests()

    def test_GRUBasic_stateSave(self):

        nNodes = 2
        nIn = 1
        maxInLen = 4 # This best be even
        batchSize = 2
        halfSequenceLengths = [maxInLen/2]*batchSize
        fullSequenceLengths = [maxInLen]*batchSize

        def getNetwork(nLayers, scopeName):
            net = nc.Network(scopeName)
            net.inputLayer(nIn)
            net.basicGRU(nNodes, nLayers=nLayers, maxSeqLen=maxInLen, batchSize=batchSize, saveState=True)
            net.buildGraph()
            gru = net.outLayer

            sess.run(tf.global_variables_initializer())
            return net, gru

        def doTests():
            # Sequence that repeats itself halfway. Running the first half twice should
            # produce the same result as running the whole sequence.
            sequence = np.array([[1],[2],[1],[2]]*batchSize)

            # These two should be different (the state should be saved from the first run)
            feed = net.getFeedDict(sequence, sequenceLengths=halfSequenceLengths)
            halfOut1, halfState1 = sess.run([gru.activations, gru.finalStates], feed_dict=feed)
            halfOut2, halfState2 = sess.run([gru.activations, gru.finalStates], feed_dict=feed)
            self.assertTrue(halfOut1.tolist() != halfOut2.tolist())
            for cellState1, cellState2 in  zip(halfState1, halfState2):
                for layerState1, layerState2 in zip(cellState1, cellState2):
                    self.assertTrue(layerState1.tolist() != layerState2.tolist())

            # Manually reset the state
            gru.resetHiddenLayer(sess)

            # The states should be the same as second run above. The outputs should be the
            # same after the zero-d outputs are removed from the end.
            feed = net.getFeedDict(sequence, sequenceLengths=fullSequenceLengths)
            fullOut, fullSeqState = sess.run([gru.activations, gru.finalStates], feed_dict=feed)
            # Reshape and trim off the ignored outputs
            fullOut = np.reshape(fullOut, [batchSize, maxInLen, nNodes])
            halfOut1 = np.reshape(halfOut1, [batchSize, maxInLen, nNodes])[:,:maxInLen/2,:]
            halfOut2 = np.reshape(halfOut2, [batchSize, maxInLen, nNodes])[:,:maxInLen/2,:]
            outconcat =  np.concatenate([halfOut1, halfOut2], axis=1)
            self.assertTrue(fullOut.tolist() == outconcat.tolist())
            for cellState1, cellState2 in  zip(fullSeqState, halfState2):
                for layerState1, layerState2 in zip(cellState1, cellState2):
                    self.assertTrue(layerState1.tolist() == layerState2.tolist())

        tf.reset_default_graph()
        with tf.Session() as sess:
            # Single Layer
            net, gru = getNetwork(nLayers=1, scopeName="singleLayer")
            doTests()
            # Multi-Layer
            net, gru = getNetwork(nLayers=3, scopeName="multiLayer")
            doTests()

    def test_GRUBasic_stateSetting(self):

        nNodes = 1
        nIn = 3
        maxInLen = 4 # This best be even
        batchSize = 2

        def getNetwork(nLayers, scopeName):
            net = nc.Network(scopeName)
            net.inputLayer(nIn)
            net.basicGRU(nNodes, nLayers=nLayers, maxSeqLen=maxInLen, batchSize=batchSize, saveState=True)
            net.buildGraph()
            gru = net.outLayer

            sess.run(tf.global_variables_initializer())
            return net, gru

        def stateToList(states, evaluate=True):
            statesList = []
            for cell in states:
                if evaluate:
                    statesList.extend([layer.eval().tolist() for layer in cell])
                else:
                    statesList.extend([layer.tolist() for layer in cell])
            return statesList

        def doTests(nLayers):
            out = net.forward(sess, inputs, sequenceLengths=maxInLen)
            stateFinal = sess.run(gru.initialStates)
            gru.resetHiddenLayer(sess)
            # After reset the state should be the zero state
            self.assertEqual(stateToList(gru.zeroStates), stateToList(gru.initialStates))
            gru.resetHiddenLayer(sess, stateFinal)
            # After resetting to value the state should eval to that value
            self.assertEqual(stateToList(stateFinal, evaluate=False), stateToList(gru.initialStates))

        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = np.random.rand(maxInLen*batchSize, nIn)

            # Single Layer
            net, gru = getNetwork(nLayers=1, scopeName="singleLayer")
            doTests(nLayers=1)

            # Multi-Layer
            nLayers = 3
            net, gru = getNetwork(nLayers, scopeName="multiLayer")
            doTests(nLayers=3)

    def test_GRUBasic_dropout(self):

        nNodes = 32
        nIn = 3
        maxInLen = 9
        batchSize = 2

        def getNetwork(nLayers, keepProb, reuse, scopeName):
            net = nc.Network(scopeName, reuse)
            net.inputLayer(nIn)
            net.basicGRU(nNodes, keepProb= keepProb, nLayers=nLayers, maxSeqLen=maxInLen,
                batchSize=batchSize, saveState=False) # Saving state defeats the purpose
            net.buildGraph()
            gru = net.outLayer

            sess.run(tf.global_variables_initializer())
            return net, gru

        def doTests():
            outNoDrop = net.forward(sess, inputs, sequenceLengths=maxInLen)
            outDrop = netDrop.forward(sess, inputs, sequenceLengths=maxInLen)
            outDrop2 = netDrop.forward(sess, inputs, sequenceLengths=maxInLen)
            # Very basic test to at least make sure that dropout changes the output values
            self.assertTrue(outNoDrop.tolist() != outDrop.tolist())
            # The dropout net likely produces a different result when run subsequently
            self.assertTrue(outDrop.tolist() != outDrop2.tolist())

        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = np.random.rand(maxInLen*batchSize, nIn)
            # Single Layer
            nLayers = 1
            net, gru = getNetwork(nLayers=1, keepProb=1.0, reuse=False, scopeName="oneLayer")
            netDrop, gruDrop = getNetwork(nLayers, keepProb=0.5, reuse=True, scopeName="oneLayer")
            doTests()

            # Multi-Layer
            nLayers = 3
            net, gru = getNetwork(nLayers, keepProb=1.0, reuse=False, scopeName="multiLayer")
            netDrop, gruDrop = getNetwork(nLayers, keepProb=0.5, reuse=True, scopeName="multiLayer")
            doTests()

    def test_concatLayer(self):
        tf.reset_default_graph()
        with tf.Session() as sess:

            nIn = 8
            nConcat = 4
            numExamples = 10

            tensorToConcat = tf.ones([numExamples, nConcat])
            tensorToAdd = tf.constant(np.random.randint(100, size=[numExamples, nConcat]))
            inputTensor = np.random.randint(100, size=[numExamples, nIn])
            expectedResult = np.concatenate([inputTensor, tensorToConcat.eval()], axis=1)

            net = nc.Network()
            net.inputLayer(nIn)
            net.concatLayer(tensorToConcat, nConcat)
            net.buildGraph()

            out = net.forward(sess, inputTensor)

            self.assertEquals(out.tolist(), expectedResult.tolist())



    def test_addLayer(self):
        tf.reset_default_graph()
        with tf.Session() as sess:

            nIn = 7
            numExamples = 10

            tensorToAdd = tf.constant(np.random.randint(100, size=[numExamples, nIn]), dtype=tf.int32)
            inputTensor = np.random.randint(100, size=[numExamples, nIn])
            expectedResult = inputTensor + tensorToAdd.eval()

            net = nc.Network()
            net.inputLayer(nIn, dtype=tf.int32)
            net.additionLayer(tensorToAdd)
            net.buildGraph()

            out = net.forward(sess, inputTensor)
            self.assertEquals(out.tolist(), expectedResult.tolist())


class TestNetwork(unittest.TestCase):

    def test_variableScope(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            net = nc.Network("Jacob")
            net.inputLayer(2)
            net.fullConnectLayer(5, None)
            net.basicGRU(4, nLayers=3, maxSeqLen=10, batchSize=2)
            net.fullConnectLayer(20, tf.nn.softmax)
            sess.run(tf.global_variables_initializer())

            self.assertEquals(net.scope.name, "Jacob")

            # Variables exist at expected scope
            with tf.variable_scope(net.scope.name + "/layer1_FullConnectLayer", reuse=True):
                fc1Weights = tf.get_variable("weights")
            with tf.variable_scope(net.scope.name + "/layer3_FullConnectLayer", reuse=True):
                fc2Weights = tf.get_variable("weights")

            # The weights retrieved via namescope are the same as those in the layers
            fc1 = net.layers[1]
            fc2 = net.outLayer
            self.assertEquals(fc1.weights.eval().tolist(), fc1Weights.eval().tolist())
            self.assertEquals(fc2.weights.eval().tolist(), fc2Weights.eval().tolist())

            # We can create a second network with the same variables, different batch size.
            net2 = nc.Network("Jacob", reuseVariables=True)
            net2.inputLayer(2)
            net2.fullConnectLayer(5, None)
            net2.basicGRU(4, nLayers=3, maxSeqLen=10, batchSize=1)
            net2.fullConnectLayer(20, tf.nn.softmax)
            sess.run(tf.global_variables_initializer())

            self.assertEquals(net2.outLayer.biases, net.outLayer.biases)

if __name__ == '__main__':
    unittest.main()
