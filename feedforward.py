import numpy as np
import tensorflow as tf


class Layer:

    def __init__(self, shape, activations, dropout=None):
        self.shape = shape
        self.activations = activations
        
        if dropout is not None:
            self.activations = tf.nn.dropout(self.activations, dropout)


class InputLayer(Layer):
    
    def __init__(self, nFeatures, dropout=None):
        shape = [None, nFeatures]
        activations = tf.placeholder(tf.float32, shape=shape)
       
        Layer.__init__(self, shape, activations, dropout)


class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, dropout, activationFunction):
        
        shape = [inLayer.shape[1], nNodes]
        self.weights = tf.Variable(tf.random_normal(shape, stddev=0.1 ))
        self.biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[nNodes]))

        weightedInput = tf.matmul(inLayer.activations, self.weights) + self.biases
        activations = activationFunction(weightedInput)
        
        Layer.__init__(self, shape, activations, dropout)

        
class FeedForwardNet:

    def __init__(self, sizes, dropouts, activations=None):

        # Default activations are ReLu for hidden, softmax for output
        if activations is None:
            activations = []
            for i in range(len(sizes)-2):
                activations.append(tf.nn.relu)
            activations.append(tf.nn.softmax)

        if len(activations) != len(sizes)-1:
            raise RuntimeError("Must specify an activation function for each layer" + \
                                 " (excluding input layer). ")    

        nIn = sizes[0]
        hiddenSizes = sizes[1:-2]
        nOut = sizes[-1]

        hiddenActivations = activations[:-2]
        outActivation = activations[-1]

        self.inLayer = InputLayer(nIn)
        self.__makeHiddenLayers__(hiddenSizes, hiddenActivations)
        self.outLayer = FullConnectLayer(self.hiddens[-1], nOut, None, outActivation)

    def __makeHiddenLayers__(self, hiddenSizes, hiddenActivations):
        
        hiddens = []
        inputs = self.inLayer
        for nNodes,activation in zip(hiddenSizes, hiddenActivations):
            hiddens.append(FullConnectLayer(inputs, nNodes, None, activation))
            inputs = hiddens[-1]          
                    
        self.hiddens = hiddens


    def forward(self, sess, x):        

        return sess.run(self.outLayer.activations, feed_dict={self.inLayer.activations:x})        


#in h1 h2 h3 ou
#d1 d2 d3 d4 
#   a1 a2 a3 a4


sess = tf.InteractiveSession()
hActivation = tf.nn.relu
outActivation = tf.nn.softmax
activations = [hActivation, hActivation, tf.nn.sigmoid, hActivation, outActivation]
net = FeedForwardNet([10,20,30,10,11,2],False, activations)
sess.run(tf.initialize_all_variables())

print net.forward(sess, np.ones([5,10]))
