import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Layer:

    def __init__(self, shape, activations, dropout=False):
        self.shape = shape
        self.activations = activations
        self.applyDropout = dropout       

        if dropout is True:
            self.keepProb = tf.placeholder(tf.float32) 
            self.activations = tf.nn.dropout(self.activations, self.keepProb)

    @staticmethod
    def __hasRequiredFields__(dictionaryKeys, requiredFields):

        for field in requiredFields:
            if field not in dictionaryKeys:
                raise RuntimeError("Missing required key, " + field  + \
                    ", in parameters dictionary")

class InputLayer(Layer):
 
    def __init__(self, nFeatures, dropout=False):
        shape = [None, nFeatures]
        activations = tf.placeholder(tf.float32, shape=shape)
       
        Layer.__init__(self, shape, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["nFeatures"]
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
    
        nFeatures = paramDic["nFeatures"]
        
        if "dropout" not in paramDic:
            dropout = False
        else:
            dropout = paramDic["dropout"]

        return cls(nFeatures, dropout)
 

class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, activationFunction, dropout=False):
        
        shape = [inLayer.shape[1], nNodes]
        self.weights = tf.Variable(tf.random_normal(shape, stddev=0.1 ))
        self.biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[nNodes]))

        weightedInput = tf.matmul(inLayer.activations, self.weights) + self.biases
        activations = activationFunction(weightedInput)
        
        Layer.__init__(self, shape, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["inLayer", "nNodes", "activationFunction"]       
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)

        inLayer = paramDic["inLayer"]
        nNodes = paramDic["nNodes"]
        activationFunction = paramDic["activationFunction"]
        
        if "dropout" not in paramDic:
            dropout = False
        else:
            dropout = paramDic["dropout"]

        return cls(inLayer, nNodes, activationFunction, dropout)


class ConvLayer(Layer):
    
    def __init__(self, inLayer, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):

        self.weights = tf.Variable(tf.truncated_normal(filterSize, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[filterSize[-1]]))

        conv = tf.nn.conv2d(inLayer.activations, self.weights, strides, padding='SAME') + \
            self.biases
        activations = activationFunction(conv)
     
        Layer.__init__(self, filterSize, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["inLayer", "activationFunction", "filterSize"]       
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
        
        inLayer = paramDic["inLayer"]
        activationFunction = paramDic["activationFunction"]
        filterSize = paramDic["filterSize"]
       
        if "strides" not in paramDic:
            strides = [1,1,1,1]
        else:
            strides = paramDic["strides"]
        if "dropout" not in paramDic:
            dropout = False
        else:
            dropout = paramDic["dropout"]

        return cls(inLayer, activationFunction, filterSize, strides, dropout)
       
        
class PoolLayer(Layer):

    def __init__(self, inLayer, poolSize, strides, dropout=False, poolType=tf.nn.max_pool):
        
        activations = poolType(inLayer.activations, ksize=poolSize, strides=strides, padding='SAME')

        Layer.__init__(self, poolSize, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["inLayer", "poolSize"]       
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
        
        inLayer = paramDic["inLayer"]
        poolSize = paramDic["poolSize"]
       
        if "strides" not in paramDic:
            strides = paramDic["poolSize"]
        else:
            strides = paramDic["strides"] 
        if "poolType" not in paramDic:
            poolType = tf.nn.max_pool
        else:
            poolType = paramDic["poolType"]
        if "dropout" not in paramDic:
            dropout = False
        else:
            dropout = paramDic["dropout"]

        return cls(inLayer, poolSize, strides, dropout, poolType)


class ReshapeLayer(Layer):

    def __init__(self, inLayer, newShape, dropout=False):
        activations = tf.reshape(inLayer.activations, newShape);
        
        Layer.__init__(self, newShape, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["inLayer", "newShape"]       
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
        
        inLayer = paramDic["inLayer"]
        newShape = paramDic["newShape"]
     
        if "dropout" not in paramDic:
            dropout = False
        else:
            dropout = paramDic["dropout"]

        return cls(inLayer, newShape,dropout)
        

class MLP():

    def __init__(self, sizes, activations=None, dropouts=None):

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


        # Input and output layers are sometimes special cases, keep them separate
        nIn = sizes[0]
        hiddenSizes = sizes[1:-2]
        nOut = sizes[-1]

        # No activations on the inputs, here we separate the output layer's activation
        hiddenActivations = activations[:-2]
        outActivation = activations[-1]

        # Define the layers
        self.inLayer = InputLayer(nIn, dropouts[0])
        self.__makeHiddenLayers__(hiddenSizes, hiddenActivations, dropouts[1:])
        self.outLayer = FullConnectLayer(self.hiddens[-1], nOut, outActivation, False)


    def __makeHiddenLayers__(self, hiddenSizes, hiddenActivations, dropouts):
        """Build a list of hidden layers.
    
        hiddenSizes -- a list of the number of nodes in each layer
        hiddenActivations -- a list of tensorflow defined activation functions for each layer
        dropouts -- a list of booleans, True when dropout is to be applied to that layer

        """

        hiddens = []
        inputs = self.inLayer
        for nNodes,activation,dropout in zip(hiddenSizes, hiddenActivations, dropouts):
            hiddens.append(FullConnectLayer(inputs, nNodes, activation, dropout))
            inputs = hiddens[-1]          
                    
        self.hiddens = hiddens


    def forward(self, sess, x, keepProb=1):        
        """Do a forward pass through the network and return the result.

        x -- The input data
        keepProb -- For layers with dropout, the probability of keeping each node

        """

        # Define the feed_dict for a forward pass
        feedDict = {self.inLayer.activations:x}
        # For each layer with dropout, add its keepProb to feed_dict
        possibleDropoutLayers = [self.inLayer]
        possibleDropoutLayers.extend(self.hiddens)
        for layer in possibleDropoutLayers:
            if layer.applyDropout:
                feedDict[layer.keepProb] = keepProb

        return sess.run(self.outLayer.activations, feed_dict=feedDict)        


class Network:

    def __init__(self, layerList):
        """
        layerList - list of tuples, each being a Layer class and a dictionary of init parameters
        """

        if len(layerList) < 3:
            raise RuntimeError('Networks with less than 3 layers not currently supported')
        
        inClass, inDictionary = layerList[0]
        self.inLayer = inClass.fromDictionary(inDictionary)

        self.layers = [self.inLayer]
        for layer in layerList[1:]:
            layerClass, layerDictionary = layer
            layerDictionary["inLayer"] = self.layers[-1]
             
            self.layers.append(layerClass.fromDictionary(layerDictionary))
            
        self.hiddens = self.layers[1:-1]
        self.outLayer = self.layers[-1]
        
    def forward(self, sess, x, keepProb=1):
        """Do a forward pass through the network and return the result.

        x -- The input data
        keepProb -- For layers with dropout, the probability of keeping each node

        """

        # Define the feed_dict for a forward pass
        feedDict = {self.inLayer.activations:x}
        # For each layer with dropout, add its keepProb to feed_dict
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

activations = (4*[hActivation]).append(outActivation)
dropouts = [True, False, False, True, True]

net = MLP([10,20,30,10,11,2], activations, dropouts)
sess.run(tf.initialize_all_variables())

print net.forward(sess, np.ones([5,10]), 0.5)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

inputLayer = InputLayer(784)
as2D = ReshapeLayer(inputLayer, [-1,28,28,1])
conv1 = ConvLayer(as2D, tf.nn.relu, [5,5,1,32])
pool1 = PoolLayer(conv1, [1, 2, 2, 1], [1, 2, 2, 1])
conv2 = ConvLayer(pool1, tf.nn.relu, [5,5,32,64])
pool2 = PoolLayer(conv2, [1, 2, 2, 1], [1, 2, 2, 1])
flat = ReshapeLayer(pool2, [-1, 7*7*64])
fc1 = FullConnectLayer(flat, 1024, tf.nn.relu, True)
readout = FullConnectLayer(fc1, 10, tf.nn.softmax)

sess.run(tf.initialize_all_variables())

print sess.run(readout.activations, feed_dict={inputLayer.activations:mnist.train.next_batch(5)[0], fc1.keepProb:1})


#Example of possible general network definition

network = Network([[InputLayer,{         "nFeatures" : 784 }], \
              [ReshapeLayer,{             "newShape" : [-1, 28, 28, 1] }], \
              [ConvLayer,{              "filterSize" : [5,5,1,32], \
                                "activationFunction" : tf.nn.relu }], \
              [PoolLayer,{                "poolSize" : [1, 2, 2, 1] }], \
              [ConvLayer,{              "filterSize" : [5,5,32,64], \
                                "activationFunction" : tf.nn.relu }], \
              [PoolLayer,{                "poolSize" : [1, 2, 2, 1] }], \
              [ReshapeLayer,{             "newShape" : [-1, 7*7*64] }], \
              [FullConnectLayer,{           "nNodes" : 1024, 
                                "activationFunction" : tf.nn.relu, \
                                           "dropout" : True }], \
              [FullConnectLayer,{           "nNodes" : 10, \
                                "activationFunction" : tf.nn.softmax }]
              ])

sess.run(tf.initialize_all_variables())
print network.forward(sess, mnist.train.next_batch(5)[0], float(0.9))

