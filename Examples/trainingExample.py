import networkExample
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
Example of how to train a network created with networkcrafter. The network is defined in 
networkExample. The architecture and this training process are from the tensorflow tutorials.
"""

#Grab the network defined in another file
network = networkExample.network

#Start a session, get the mnist data
sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Placeholder for the target values
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Cost function, training, and metrics.
cross_entropy = -tf.reduce_sum(y_*tf.log(network.outLayer.activations))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(network.outLayer.activations, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(200):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    #Get the network to build a feed dictionary. Includes inputs, dropouts, targets.
    feedDict = network.getFeedDict(inputs=batch[0], keepProb=1.0, extras={y_: batch[1]})
    
    train_accuracy = accuracy.eval(feed_dict=feedDict)
    print("step %d, training accuracy %g"%(i, train_accuracy))
  
  feedDict = network.getFeedDict(inputs=batch[0], keepProb=0.5, extras={y_:batch[1]})
  train_step.run(feed_dict=feedDict)

#Test set accuracy
feedDict = network.getFeedDict(mnist.test.images, 1.0, {y_ : mnist.test.labels})
print("test accuracy %g"%accuracy.eval(feed_dict=feedDict))
