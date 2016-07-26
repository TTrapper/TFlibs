import tensorflow as tf
import numpy as np
import networkcrafter as nc

sess = tf.InteractiveSession()

data = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])

network = nc.Network()
network.inputLayer(3)
#network.rnnLayer(2)
network.tfRNN(2)

sess.run(tf.initialize_all_variables())

print network.forward(sess, data)
one = data[0,:]
one.shape = 1,3
network.resetRecurrentHiddens(sess)
print network.forward(sess, one)
print network.forward(sess, data)
