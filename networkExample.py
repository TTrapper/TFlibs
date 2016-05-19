import networkcrafter as nc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
Example of how to use networkcrafter to design arbitrary networks, combining many Layer types.
"""
# networkcrafter is built on tensorflow, here we start a tf session
sess = tf.InteractiveSession()

# Some tensorflow provided mnist data for our convolutional neural net
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
Example of possible general network definition. The Network constructor takes a list of tuples.
Each tuple consists of a Layer class and a dictionary of initialization parameters. Default
values are automatically filled in by the layers when they are initialized. 
"""
network = nc.Network([[nc.InputLayer,{     "nFeatures" : 784 }], \
                 [nc.ReshapeLayer,{         "newShape" : [-1, 28, 28, 1] }], \
                 [nc.ConvLayer,{          "filterSize" : [5,5,1,32], \
                                  "activationFunction" : tf.nn.relu }], \
                 [nc.PoolLayer,{            "poolSize" : [1, 2, 2, 1] }], \
                 [nc.ConvLayer,{          "filterSize" : [5,5,32,64], \
                                  "activationFunction" : tf.nn.relu }], \
                 [nc.PoolLayer,{            "poolSize" : [1, 2, 2, 1] }], \
                 [nc.ReshapeLayer,{         "newShape" : [-1, 7*7*64] }], \
                 [nc.FullConnectLayer,{       "nNodes" : 1024,
                                  "activationFunction" : tf.nn.relu, \
                                             "dropout" : True }], \
                 [nc.FullConnectLayer,{       "nNodes" : 10, \
                                  "activationFunction" : tf.nn.softmax }]
                 ])
sess.run(tf.initialize_all_variables())

# Do a forward pass through the network. For layers with dropout the keep_prob is 0.9
print network.forward(sess, mnist.train.next_batch(5)[0], float(0.9))
