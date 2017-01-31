import unittest
import TFlibs.networkcrafter as nc
import tensorflow as tf
import numpy as np

def testDropout(layer, expectDropoutOn):
   if expectDropoutOn:
        return layer.applyDropout and (type(layer.keepProb) is tf.Tensor)
   else:
        return (not layer.applyDropout) and (layer.keepProb is None)

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
            self.assertTrue(layer.keepProb is None)
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


    def test_GRUBasic_stateOut(self):

        nNodes = 2
        nIn = 4
        maxInLen = 10
        inputs = [1]*maxInLen
        inLength = 5

        def getNetwork(nLayers):
            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True)
            net.basicGRU(nNodes, nLayers=nLayers,  maxSeqLen=maxInLen, saveState=False, activationsAreFinalState=True)
            net.buildGraph()
            gru = net.outLayer
            return net, gru

        def doTests():

            sess.run(tf.global_variables_initializer())

            feed = net.getFeedDict(inputs, sequenceLengths=inLength)
            finalState, outSequence = sess.run([net.outputs, gru.outSequence], feed)
            # Timestep reshaped to dim-0
            outSequence = np.reshape(outSequence, [-1, nNodes])

            # network output is the final state which is the output at time inLength
            self.assertTrue(outSequence[inLength-1, :].tolist() == finalState[0].tolist())

        tf.reset_default_graph()
        with tf.Session() as sess:
            with tf.variable_scope("singleLayer"):
                net, gru = getNetwork(1)
            doTests()

            with tf.variable_scope("multiLayer"):
                net, gru = getNetwork(2)
            doTests()

    def test_GRUBasic_stateSave(self):

        nNodes = 2
        nIn = 1
        maxInLen = 4 # This best be even
        batchSize = 2
        halfSequenceLengths = [maxInLen/2]*batchSize
        fullSequenceLengths = [maxInLen]*batchSize

        def getNetwork(nLayers):
            net = nc.Network()
            net.inputLayer(nIn)
            net.basicGRU(nNodes, nLayers=nLayers, maxSeqLen=maxInLen, batchSize=batchSize, saveState=True)
            net.buildGraph()
            gru = net.outLayer

            return net, gru

        def doTests():
            # Sequence that repeats itself halfway. Running the first half twice should
            # produce the same result as running the whole sequence.
            sequence = np.array([[1],[2],[1],[2]]*batchSize)

            # These two should be different (the state should be saved from the first run)
            feed = net.getFeedDict(sequence, sequenceLengths=halfSequenceLengths)
            halfOut1, halfState1 = sess.run([gru.activations, gru.state], feed_dict=feed)
            halfOut2, halfState2 = sess.run([gru.activations, gru.state], feed_dict=feed)
            self.assertTrue(halfOut1.tolist() != halfOut2.tolist())
            if (gru.nLayers == 1):
                self.assertTrue(halfState1.tolist() != halfState2.tolist())

            # Manually reset the state
            gru.resetHiddenLayer(sess)

            # The states should be the same as second run above. The outputs should be the
            # same after the zero-d outputs are removed from the end.
            feed = net.getFeedDict(sequence, sequenceLengths=fullSequenceLengths)
            fullOut, fullSeqState = sess.run([gru.activations, gru.state], feed_dict=feed)
            # Reshape and trim off the ignored outputs
            fullOut = np.reshape(fullOut, [batchSize, maxInLen, nNodes])
            halfOut1 = np.reshape(halfOut1, [batchSize, maxInLen, nNodes])
            halfOut2 = np.reshape(halfOut2, [batchSize, maxInLen, nNodes])
            halfOut1 = np.reshape(halfOut1, [batchSize, maxInLen, nNodes])[:,:maxInLen/2,:]
            halfOut2 = np.reshape(halfOut2, [batchSize, maxInLen, nNodes])[:,:maxInLen/2,:]
            outconcat =  np.concatenate([halfOut1, halfOut2], axis=1)
            self.assertTrue(fullOut.tolist() == outconcat.tolist())
            if (gru.nLayers == 1):
                self.assertTrue(fullSeqState.tolist() == halfState2.tolist())

        tf.reset_default_graph()
        with tf.Session() as sess:
            # Single Layer
            with tf.variable_scope("netSingleLayer"):
                net, gru = getNetwork(nLayers=1)
            sess.run(tf.global_variables_initializer())
            doTests()
            # Multi-Layer
            with tf.variable_scope("netMultiLayer"):
                net, gru = getNetwork(nLayers=3)
            sess.run(tf.global_variables_initializer())
            doTests()

    def test_seq2SeqBasic_feedPrev(self):

        tf.reset_default_graph()
        with tf.Session() as sess:

            nNodes = 3
            nOut = 2
            nIn = 5
            outLen = 10
            inLen = 3

            readout = nc.Network()
            readout.inputLayer(nNodes)
            readout.fullConnectLayer(nOut, None)

            encodeIn = nc.InputLayer(nIn, applyOneHot=True)
            decodeIn = nc.InputLayer(nOut, applyOneHot=True)

            seq2seq = nc.Seq2SeqBasic(encodeIn, decodeIn, nNodes, inLen, outLen, readout)
            seq2seq.buildGraph()

            readout.buildGraph()

            enData = np.ones([inLen])
            deData = 2*np.ones([outLen])

            sess.run(tf.global_variables_initializer())

            feed = {encodeIn.inputs:enData, decodeIn.inputs:deData, seq2seq.enSequenceLengths:[3]}
            outFeedDecode = sess.run(seq2seq.activations, feed_dict=feed).tolist()
            seq2seq.setFeedPrevious(True, sess)
            outFeedPrev = sess.run(seq2seq.activations, feed_dict=feed).tolist()
            readoutFeedDecode = sess.run(readout.outLayer.activations, feed_dict=feed).tolist()


            self.assertFalse(outFeedDecode == outFeedPrev)



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



if __name__ == '__main__':
    unittest.main()
