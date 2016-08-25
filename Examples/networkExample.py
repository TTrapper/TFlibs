import TFlibs.networkcrafter as nc
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
Example of how to use networkcrafter to design arbitrary networks, combining many Layer types.
"""

# Possible general network definition.
network = nc.Network()
network.inputLayer(nFeatures=784)
network.reshapeLayer(newShape=[-1, 28, 28, 1])
network.convLayer(filterSize=[5, 5, 1, 32], activationFunction=tf.nn.relu)
network.poolLayer(poolSize=[1, 2, 2, 1])
network.convLayer(filterSize=[5, 5, 32, 64], activationFunction=tf.nn.relu)
network.poolLayer(poolSize=[1, 2, 2, 1])
network.reshapeLayer(newShape=[-1, 7*7*64])
network.fullConnectLayer(nNodes=1024, activationFunction=tf.nn.relu, dropout=True)
network.fullConnectLayer(nNodes=10, activationFunction=tf.nn.softmax)


# Do a forward pass through the network. For layers with dropout the keep_prob is 0.9
sess.run(tf.initialize_all_variables())
print network.forward(sess, mnist.train.next_batch(5)[0], float(0.9))

print network.outLayer.weights.eval(session=sess)
