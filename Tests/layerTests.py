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

    def test_seq2SeqBasic_feedPrev(self):

        nNodes = 100
        nOut = 2
        nIn = 5
        outLen = 4
        inLen = 3

        encodeIn = nc.InputLayer(nIn, applyOneHot=True)
        decodeIn = nc.InputLayer(nOut, applyOneHot=True)
        wb = nc.FullConnectLayer.xavierInit([nNodes, nOut])

        seq2seq = nc.Seq2SeqBasic(encodeIn, decodeIn, nNodes, inLen, outLen, wb)
        seq2seq.buildGraph()

        readout = nc.FullConnectLayer(seq2seq, nOut, None, wb=wb)
        readout.buildGraph()

        enData = np.ones([inLen])
        deData = np.ones([outLen])

        sess.run(tf.initialize_all_variables())

        feed = {encodeIn.inputs:enData, decodeIn.inputs:deData}
        outFeedDecode = sess.run(seq2seq.activations, feed_dict=feed).tolist()
        seq2seq.setFeedPrevious(True, sess)
        outFeedPrev = sess.run(seq2seq.activations, feed_dict=feed).tolist()
        readoutFeedDecode = sess.run(readout.activations, feed_dict=feed).tolist()

        self.assertFalse(outFeedDecode == outFeedPrev)



if __name__ == '__main__':
    with tf.Session() as sess:
        unittest.main()
