import unittest
import networkcrafter as nc
import tensorflow as tf
import numpy as np

class TestLayerInitialization(unittest.TestCase):

    def test_base_layer_init(self):
        # Layer with no dropout 
        shape = [10, 4]
        layer = nc.Layer(shape, tf.nn.softmax(np.ones([4,1])))
        self.assertEqual(shape, layer.shape)

        self.assertTrue(isinstance(layer.activations, tf.Tensor))
        self.assertFalse(layer.applyDropout)
        with self.assertRaises(AttributeError):
            self.keepProb
        with self.assertRaises(TypeError):
            layer = nc.Layer(shape, tf.nn.softmax, True)


        # Layer with dropout
        shape = [4, 10]
          
        layer = nc.Layer(shape, tf.nn.softmax(tf.placeholder(tf.float32, shape=[10,1])), True)
        self.assertEqual(shape, layer.shape) 
        self.assertTrue(layer.applyDropout)
        self.assertTrue(layer.keepProb)

#    def test_input_layer_init(self):

        


if __name__ == '__main__':
    unittest.main()
