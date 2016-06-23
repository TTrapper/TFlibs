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
        activations = tf.placeholder(tf.float32, shape=shape, name='xin')
       
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

def loopTry0(x):

    c = lambda i: tf.less(tf.reduce_sum(i), 100)
    b = lambda i: tf.add(i, 1)
    return tf.while_loop(c,b,[x])


def loopTry1(x):

    y = tf.ones([1,4])
    def body(q, z):

        v = tf.add(z, 1)
        z = tf.add(v,v)
 
        return tf.add(q,1), z 

    c = lambda i, j: tf.less(tf.reduce_sum(i), 16)

    b = body 
    return tf.while_loop(c,b,[x, y])


def loopTry(x):

    y = tf.ones([1,2], dtype=tf.int32)
    co = tf.constant(0)
    def body(q, z, counter):

        v = tf.add(z, 1)
#        z = tf.add(v,v)
#        z = tf.slice(q, [counter, 0], size=[1,2])
        z = tf.slice(q, [counter, 0], [1,2]) 
        counter = tf.add(counter, 1) 
        return tf.add(q,1), z, counter

    c = lambda i, j, k: tf.less(k, 3)

    b = body 
    return tf.while_loop(c,b,[x, y, co])


class RNN(Layer):

    def __init__(self, inLayer, nNodes, nOut, activationFunction, dropout=False):
        
        self.inLayer = inLayer

        self.xWeights = tf.Variable(tf.truncated_normal([inLayer.shape[1], nNodes], stddev=0.1), name='xW')
        self.hWeights = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1), name='hW')
        self.yWeights = tf.Variable(tf.truncated_normal([nNodes, nOut], stddev=0.1), name='yW')

        self.xBias = tf.Variable(tf.zeros([nNodes]))
        self.hBias = tf.Variable(tf.zeros([nNodes]))
        self.yBias = tf.Variable(tf.zeros([nOut]))

        self.h = tf.zeros([1, nNodes])

        self.index = tf.constant(0, dtype=tf.int32)   
 
        loopParams = [self.index,\
                      self.xWeights,\
                      self.xBias, \
                      self.hWeights, \
                      self.hBias, \
                      self.yWeights, \
                      self.yBias, \
                      inLayer.activations, \
                      self.h, \
                      tf.zeros([1, inLayer.shape[1]], dtype=tf.float32)]

        def updateLoopBody(idx, xW, xB, hW, hB, yW, yB, x, h, y):

            x_t = tf.slice(x, [idx, 0], [1, -1])

            uX = tf.matmul(x_t, xW) + xB
            uH = tf.matmul(h, hW) + hB
            h = tf.tanh(uX + uH)

            y_t = tf.matmul(h, yW) + yB

            y = tf.concat(0, [y, y_t])
            idx = tf.add(idx, 1)

            return idx, xW, xB, hW, hB, yW, yB, x, h, y
           # return idx, xW, xB, hW, hB, yW, yB, x, h, tf.slice(y, [1,0], [-1, -1])

        condition = lambda idx, xW, xB, hW, hB, yW, yB, x, h, y: tf.less(idx, tf.shape(x)[0])
        updateLoop = tf.while_loop(condition, updateLoopBody, loopParams)

        activations = activationFunction(tf.slice(updateLoop[-1], [1,0], [-1, -1]))

        #activations = activationFunction(updateLoop[-1])
        """
        # Hidden state
        self.h = tf.placeholder(tf.float32, shape=[1, nNodes], name='hS')
        
        # Define the update to the hidden state        
        self.updateX = tf.matmul(inLayer.activations, self.xWeights) + self.xBias
        self.updateH = tf.matmul(self.h, self.hWeights) + self.hBias;
        self.newH = tf.tanh(self.updateX + self.updateH)

        activations = activationFunction(tf.matmul(self.newH, self.yWeights) + self.yBias)
        """
        Layer.__init__(self, [nNodes, nOut], activations, dropout)
   
#    def getZeroState(self):
#        return np.zeros([1, self.shape[0]])

#    def getCurrentState(self, x, h, sess):
#        return self.newH.eval(feed_dict={self.inLayer.activations:x, self.h:h}, session=sess)


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
