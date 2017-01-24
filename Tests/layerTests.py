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
            sess.run(tf.initialize_all_variables())

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
            sess.run(tf.initialize_all_variables())

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
            sess.run(tf.initialize_all_variables())

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
        tf.reset_default_graph()
        with tf.Session() as sess:
            nNodes = 2
            nIn = 4
            maxInLen = 10

            net = nc.Network()
            net.inputLayer(nIn, applyOneHot=True)
            net.basicGRU(nNodes, maxSeqLen=maxInLen, saveState=False, activationsAreFinalState=True)
            net.buildGraph()

            sess.run(tf.initialize_all_variables())
            inputs = [1]*maxInLen
            inLength = 5

            feed = net.getFeedDict(inputs, sequenceLengths=inLength)
            state, outs = sess.run([net.outputs, net.layers[1].outputs], feed)
            # Timestep reshaped to dim-0
            outs = np.reshape(outs, [-1, nNodes])

            # The final state is not updated past the sequence length (inLength)
            self.assertTrue(outs[inLength-1, :].tolist() == state[0].tolist())


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

            sess.run(tf.initialize_all_variables())

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
