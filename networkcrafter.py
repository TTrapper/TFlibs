import numpy as np
import tensorflow as tf

class Layer:

    def __init__(self, shape, activations, dropout=False):
        """
        shape -- the shape of the weights on this layer
        activations -- the output values of this layer, may be a tf.placeholder or tensor
        dropout -- whether or not dropout is to be applied to this layer
        """

        if not isinstance(activations, tf.Tensor):
            raise TypeError("A layer's activations must be of type TensorFlow.Tensor.")

        self.shape = shape
        self.activations = activations

        self.applyDropout = dropout       

        if dropout is True:
            self.keepProb = tf.placeholder(tf.float32) 
            self.activations = tf.nn.dropout(self.activations, self.keepProb)


class InputLayer(Layer):
 
    def __init__(self, nFeatures, dropout=False):
        shape = [None, nFeatures]
        activations = tf.placeholder(tf.float32, shape=shape)
       
        Layer.__init__(self, shape, activations, dropout)


class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, activationFunction, dropout=False):
        
        shape = [inLayer.shape[1], nNodes]
        self.weights = tf.Variable(tf.random_normal(shape, stddev=0.1 ))
        self.biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[nNodes]))

        weightedInput = tf.matmul(inLayer.activations, self.weights) + self.biases
        activations = activationFunction(weightedInput)
        
        Layer.__init__(self, shape, activations, dropout)


class ConvLayer(Layer):
    
    def __init__(self, inLayer, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):

        self.weights = tf.Variable(tf.truncated_normal(filterSize, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[filterSize[-1]]))

        conv = tf.nn.conv2d(inLayer.activations, self.weights, strides, padding='SAME') + \
            self.biases
        activations = activationFunction(conv)
     
        Layer.__init__(self, filterSize, activations, dropout)

        
class PoolLayer(Layer):

    def __init__(self, inLayer, poolSize, strides, dropout=False, poolType=tf.nn.max_pool):
        
        activations = poolType(inLayer.activations, ksize=poolSize, strides=strides, padding='SAME')

        Layer.__init__(self, poolSize, activations, dropout)


class ReshapeLayer(Layer):

    def __init__(self, inLayer, newShape, dropout=False):
        activations = tf.reshape(inLayer.activations, newShape);
        
        Layer.__init__(self, newShape, activations, dropout)


class RNN(Layer):

    def __init__(self, inLayer, nNodes, nOut, activationFunction, dropout=False):
        
        self.xWeights = tf.Variable(tf.truncated_normal([inLayer.shape[1], nNodes], stddev=0.1))
        self.hWeights = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        self.yWeights = tf.Variable(tf.truncated_normal([nNodes, nOut], stddev=0.1))

        self.xBias = tf.Variable(tf.zeros([nNodes]))
        self.hBias = tf.Variable(tf.zeros([nNodes]))
        self.yBias = tf.Variable(tf.zeros([nOut]))

        # Hidden state
        self.h = tf.zeros([1, nNodes], tf.float32)

        # Define the update to the hidden state        
        updateX = tf.matmul(inLayer.activations, self.xWeights) + self.xBias
        updateH = tf.matmul(self.h, self.hWeights) + self.hBias;
        self.h = tf.tanh(0.5*updateX + updateH)

        activations = activationFunction(tf.matmul(self.h, self.yWeights) + self.yBias)

        Layer.__init__(self, [nNodes, nOut], activations, dropout)
   
    def resetHiddenState(self):
        self.h = tf.zeros([1, self.shape[0]], tf.float32)
     

class Network:

    def inputLayer(self, nFeatures, dropout=False):
        self.inLayer = InputLayer(nFeatures, dropout)
        self.layers = [self.inLayer]
        self.hiddens = []
        self.outLayer = self.inLayer
      
    def fullConnectLayer(self,  nNodes, activationFunction, dropout=False):
        self.__addLayer__(FullConnectLayer(self.outLayer, nNodes, activationFunction, dropout))

    def convLayer(self, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):
        self.__addLayer__(ConvLayer(self.outLayer, activationFunction, filterSize, strides, dropout))
    
    def poolLayer(self,  poolSize, strides=None, dropout=False, poolType=tf.nn.max_pool):
        if strides is None:
            strides = poolSize
        self.__addLayer__(PoolLayer(self.outLayer, poolSize, strides, dropout, poolType))

    def reshapeLayer(self, newShape, dropout=False):
        self.__addLayer__(ReshapeLayer(self.outLayer, newShape, dropout))
    
    def __addLayer__(self, layer):
        self.layers.append(layer)
        self.hiddens.append(layer)
        self.outLayer = layer

    def __init__(self):
        self.hiddens =[]
        
    def forward(self, sess, inputs, keepProb=1):
        """Do a forward pass through the network and return the result.

        inputs -- The input data
        keepProb -- For layers with dropout, the probability of keeping each node

        """
        feedDict = self.getFeedDict(inputs, keepProb)

        return sess.run(self.outLayer.activations, feed_dict=feedDict)        
        
    def getFeedDict(self, inputs, keepProb, extras={}):

        """Create a feed dicionary for tensorflow. 

        inputs -- The input data to be slotted into the input layer activations
        keepProb -- For layers with dropout, the probability of keeping each node
        extras -- Any values to be added to the dictionary that are external to the network
        """
        # Define the feed_dict for a forward pass
        feedDict = {self.inLayer.activations : inputs}
        # For each layer with dropout, add its keepProb to feed_dict
        possibleDropoutLayers = [self.inLayer]
        possibleDropoutLayers.extend(self.hiddens)
        for layer in possibleDropoutLayers:
            if layer.applyDropout:
                feedDict[layer.keepProb] = keepProb

        feedDict.update(extras)
        return feedDict

class MLP(Network):

    def __init__(self, sizes, activations=None, dropouts=None):
        """
        sizes -- list: number of nodes in each layer, including input features and readout layer
        activations -- list: activation function for each layer (excluding input layer)
        dropouts -- list: boolean, whether or not to apply dropout for each layer (excluding input)
        """

        # Default activations are ReLu for hidden, softmax for output
        if activations is None:
            activations = [tf.nn.relu]*(len(sizes)-2)
            activations.append(tf.nn.softmax)

        if len(activations) != len(sizes)-1:
            raise RuntimeError("Must specify an activation function for each layer" + \
                                 " (excluding input layer). Use None for default.")    

        # By default, do not apply dropout to any layer
        if dropouts is None:
            dropouts = [False]*(len(sizes)-1)

        if len(dropouts) != len(sizes)-1:
            raise RuntimeError("Must specify whether to apply dropout to each layer" + \
                                " (excluding output layer). Use None for default.") 
        
        # Never use dropout on the last layer
        dropouts.append(False)       
 
        # Initialize superclass and add the layers
        Network.__init__(self)
        
        self.inputLayer(sizes[0], dropouts[0])
        for nNodes,activation,dropout in zip(sizes[1:], activations, dropouts[1:]):
            self.fullConnectLayer(nNodes, activation, dropout)
