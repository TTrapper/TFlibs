import numpy as np
import tensorflow as tf

class Layer:

    def __init__(self, shape, activations, dropout=False):
        """
        shape -- the shape of the weights on this layer
        activations -- the output values of this layer, may be a tf.placeholder or tensor
        dropout -- whether or not dropout is to be applied to this layer
        """
        self.shape = shape
        self.activations = activations
        self.applyDropout = dropout       

        if dropout is True:
            self.keepProb = tf.placeholder(tf.float32) 
            self.activations = tf.nn.dropout(self.activations, self.keepProb)

    @staticmethod
    def __hasRequiredFields__(dictionaryKeys, requiredFields):
        """Used by subclasses to ensure that parameter-value dictionaries have necessary values
       
        dictionaryKeys -- keys that exist in a dictionary
        requiredFields -- keys that are expected to exist in the dictionary
        """

        for field in requiredFields:
            if field not in dictionaryKeys:
                raise RuntimeError("Missing required key, " + field  + \
                    ", in parameters dictionary")

    @staticmethod
    def __updateDefaults__(initDictionary, layerDefaults={}):
        """Used by subclasses to fill in missing init parameters with defaults. User specified
        values take precedence over the layers defaults, which take precedence over globals.

        initDictionary -- User supplied dictionary of parameters
        layerDefaults -- default values specific to the Layer subclass
        """
  
        globalDefaults = {"dropout" : False}
        
        globalDefaults.update(layerDefaults)
        globalDefaults.update(initDictionary)
        
        return globalDefaults        


class InputLayer(Layer):
 
    def __init__(self, nFeatures, dropout=False):
        shape = [None, nFeatures]
        activations = tf.placeholder(tf.float32, shape=shape)
       
        Layer.__init__(self, shape, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["nFeatures"]
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
    
        paramDic = Layer.__updateDefaults__(paramDic)

        nFeatures = paramDic["nFeatures"]
        dropout   = paramDic["dropout"]

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
      
        paramDic = Layer.__updateDefaults__(paramDic)

        inLayer            = paramDic["inLayer"]
        nNodes             = paramDic["nNodes"]
        activationFunction = paramDic["activationFunction"]       
        dropout            = paramDic["dropout"]

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
        
        layerDefaults = {"strides":[1,1,1,1]}
        paramDic = Layer.__updateDefaults__(paramDic, layerDefaults)

        inLayer            = paramDic["inLayer"]
        activationFunction = paramDic["activationFunction"]
        filterSize         = paramDic["filterSize"]  
        strides            = paramDic["strides"]
        dropout            = paramDic["dropout"]

        return cls(inLayer, activationFunction, filterSize, strides, dropout)
       
        
class PoolLayer(Layer):

    def __init__(self, inLayer, poolSize, strides, dropout=False, poolType=tf.nn.max_pool):
        
        activations = poolType(inLayer.activations, ksize=poolSize, strides=strides, padding='SAME')

        Layer.__init__(self, poolSize, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["inLayer", "poolSize"]       
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
        
        layerDefaults = {"strides" : paramDic["poolSize"], "poolType" : tf.nn.max_pool}
        paramDic = Layer.__updateDefaults__(paramDic, layerDefaults)

        inLayer  = paramDic["inLayer"]
        poolSize = paramDic["poolSize"]     
        strides  = paramDic["strides"] 
        poolType = paramDic["poolType"]
        dropout  = paramDic["dropout"]

        return cls(inLayer, poolSize, strides, dropout, poolType)


class ReshapeLayer(Layer):

    def __init__(self, inLayer, newShape, dropout=False):
        activations = tf.reshape(inLayer.activations, newShape);
        
        Layer.__init__(self, newShape, activations, dropout)

    @classmethod
    def fromDictionary(cls, paramDic):

        requiredFields = ["inLayer", "newShape"]       
        Layer.__hasRequiredFields__(paramDic.keys(), requiredFields)
        
        paramDic = Layer.__updateDefaults__(paramDic)

        inLayer  = paramDic["inLayer"]
        newShape = paramDic["newShape"]
        dropout  = paramDic["dropout"]
        
        return cls(inLayer, newShape,dropout)


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


        # Input and output layers are sometimes special cases, keep them separate
        nIn = sizes[0]
        hiddenSizes = sizes[1:-2]
        nOut = sizes[-1]

        # No activations on the inputs, here we separate the output layer's activation
        hiddenActivations = activations[:-2]
        outActivation = activations[-1]

        # Define the layers as a list of tuples to be passed to Network constructor
        initList = [[InputLayer, {"nFeatures" : nIn, "dropout" : dropouts[0]}]] 

        for nNodes,activation,dropout in zip(hiddenSizes, hiddenActivations, dropouts[1:]):
            initList.append([FullConnectLayer, { "nNodes" : nNodes,\
                                     "activationFunction" : activation,\
                                                "dropout" : dropout}])

        initList.append([FullConnectLayer, { "nNodes" : nOut,\
                                 "activationFunction" : outActivation}])

        Network.__init__(self, initList)