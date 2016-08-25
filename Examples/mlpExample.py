import TFlibs.networkcrafter as nc
import numpy as np
import tensorflow as tf

"""
Example of how to create an Multi-Layer-Perceptron with networkcrafter.
"""

# networkcrafter is built on tensorflow, here we start a tf session
sess = tf.InteractiveSession()

# Define some activation functions for the hidden (rectified linear) and output (softmax) layers.
activations = (4*[tf.nn.relu]).append(tf.nn.softmax)

# Each layer, excluding the in/output, can have dropout applied. Note that the inputs are a layer. 
dropouts = [False, False, True, True]

# Define the number of nodes for each layer, including input features and readout nodes
layerSizes = [10,20,30,10,11,2]

# Initialize the MLP netowrk
net = nc.MLP(layerSizes, activations, dropouts)
net.buildGraph()
sess.run(tf.initialize_all_variables())

# Do a forward pass through the network. For layers with dropout keep_prob is 0.5.
print net.forward(sess, np.ones([5,10]), 0.5)
