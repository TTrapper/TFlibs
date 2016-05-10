import numpy as np
import tensorflow as tf

class Layer:

    def __init__(self, shape, activations, dropout=False):
        self.shape = shape
        self.activations = activations
        self.applyDropout = dropout       

        if dropout is True:
            self.keepProb = tf.placeholder(tf.float32) 
            self.activations = tf.nn.dropout(self.activations, self.keepProb)


class InputLayer(Layer):
    
    def __init__(self, nFeatures, dropout=None):
        shape = [None, nFeatures]
        activations = tf.placeholder(tf.float32, shape=shape)
       
        Layer.__init__(self, shape, activations, dropout)


class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, activationFunction, dropout):
        
        shape = [inLayer.shape[1], nNodes]
        self.weights = tf.Variable(tf.random_normal(shape, stddev=0.1 ))
        self.biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[nNodes]))

        weightedInput = tf.matmul(inLayer.activations, self.weights) + self.biases
        activations = activationFunction(weightedInput)
        
        Layer.__init__(self, shape, activations, dropout)

        
class MLP:

    def __init__(self, sizes, activations=None, dropouts=None):

        # Default activations are ReLu for hidden, softmax for output
        if activations is None:
            activations = [tf.nn.relu]*(len(sizes)-2)
            activations.append(tf.nn.softmax)

        if len(activations) != len(sizes)-1:
            raise RuntimeError("Must specify an activation function for each layer" + \
                                 " (excluding input layer). Use None for default.")    

        if dropouts is None:
            dropouts = [False]*(len(sizes)-1)

        if len(dropouts) != len(sizes)-1:
            raise RuntimeError("Must specify whether to apply dropout to each layer" + \
                                " (excluding output layer). Use None for default.")

        nIn = sizes[0]
        hiddenSizes = sizes[1:-2]
        nOut = sizes[-1]

        hiddenActivations = activations[:-2]
        outActivation = activations[-1]

        self.inLayer = InputLayer(nIn, dropouts[0])
        self.__makeHiddenLayers__(hiddenSizes, hiddenActivations, dropouts[1:])
        self.outLayer = FullConnectLayer(self.hiddens[-1], nOut, outActivation, False)

    def __makeHiddenLayers__(self, hiddenSizes, hiddenActivations, dropouts):
        
        hiddens = []
        inputs = self.inLayer
        for nNodes,activation,dropout in zip(hiddenSizes, hiddenActivations, dropouts):
            hiddens.append(FullConnectLayer(inputs, nNodes, activation, dropout))
            inputs = hiddens[-1]          
                    
        self.hiddens = hiddens


    def forward(self, sess, x, keepProb=1):        

        feedDict = {self.inLayer.activations:x}
        
        possibleDropoutLayers = [self.inLayer]
        possibleDropoutLayers.extend(self.hiddens)
        for layer in possibleDropoutLayers:
            if layer.applyDropout:
                feedDict[layer.keepProb] = keepProb

        return sess.run(self.outLayer.activations, feed_dict=feedDict)        


#in h1 h2 h3 ou
#d1 d2 d3 d4 
#   a1 a2 a3 a4

sess = tf.InteractiveSession()
hActivation = tf.nn.relu
outActivation = tf.nn.softmax

activations = [hActivation, hActivation, hActivation , hActivation, outActivation]
dropouts = [True, False, False, True, True]

net = MLP([10,20,30,10,11,2], activations, dropouts)
sess.run(tf.initialize_all_variables())

print net.forward(sess, np.ones([5,10]), 0.5)

