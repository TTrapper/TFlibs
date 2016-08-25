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
        # Layer with no dropout
        shape = [10, 4]
        layer = nc.Layer(shape, tf.nn.softmax(np.ones([4,1])))

        self.assertEqual(shape, layer.shape)
        self.assertTrue(isinstance(layer.activations, tf.Tensor))
        self.assertTrue(testDropout(layer, expectDropoutOn=False))
        with self.assertRaises(TypeError):
            layer = nc.Layer(shape, tf.nn.softmax, True)

        # Layer with dropout
        shape = [4, 10]
        layer = nc.Layer(shape, tf.ones(shape=[10,1], dtype=tf.float32), tf.nn.softmax, True)

        self.assertEqual(shape, layer.shape)
        self.assertTrue(isinstance(layer.activations, tf.Tensor))
        self.assertTrue(testDropout(layer, expectDropoutOn=True))

    def test_input_layer(self):
        nFeatures = 10
        setShape = [2, nFeatures]
        layer = nc.InputLayer(nFeatures, applyOneHot=False, dtype=tf.int32)
        feed = {layer.inputs:np.ones(shape=setShape)}

        self.assertTrue(layer.shape == [None, nFeatures])
        self.assertTrue(testDropout(layer, False))
        shape = tf.shape(layer.activations).eval(session=sess, feed_dict=feed)
        self.assertTrue(shape.all() == np.array(setShape).all())




if __name__ == '__main__':
    with tf.Session() as sess:
        unittest.main()
