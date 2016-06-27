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


class RNN(Layer):

    def __init__(self, inLayer, nNodes, dropout=False):
        
        self.inLayer = inLayer
        
        # Weight matrices. x: input hidden, h: recurrent hidden, y: output 
        self.xW = tf.Variable(tf.truncated_normal([inLayer.shape[1], nNodes], stddev=0.1))
        self.hW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))

        # Biases
        self.xB = tf.Variable(tf.zeros([nNodes]))
        self.hB = tf.Variable(tf.zeros([nNodes]))
        
        # Hidden state/memory
        self.h = tf.Variable(tf.zeros([1, nNodes]))

        # Using trainable weights, get a new representation of inputs with same dimension as h
        xTransform = tf.matmul(inLayer.activations, self.xW) + self.xB

        # Loop counter
        self.index = tf.constant(0, dtype=tf.int32) 
        loopParams = [self.index,\
                      xTransform, \
                      self.h, \
                      tf.zeros([tf.shape(inLayer.activations)[0], nNodes], dtype=tf.float32)]
                      
        # Each iteration performs one step of RNN update, produces series of hidden states
        def updateLoopBody(idx, x, h, activations):
            # Grab weighted representation of the current input
            x_t = tf.slice(x, [idx, 0], [1, -1])
            
            # Update the hidden state with recurrent connections and inputs
            uH = tf.matmul(h, self.hW) + self.hB
            h = tf.tanh(x_t + uH) 

            # This is an awkward way of getting activations to have the same shape as the targets.
            def firstIteration(): return h
            def nextIterations(): return tf.concat(0, [activations, h])
            activations = tf.cond(tf.equal(0, idx), firstIteration, nextIterations)
    
            idx = tf.add(idx, 1)

            return idx, x, h, activations

        # The update loop runs for each example in the batch.
        condition = lambda idx, x, h, activations: tf.less(idx, tf.shape(x)[0])
        updateLoop = tf.while_loop(condition, updateLoopBody, loopParams)

        # A time series of the RNN's hidden state accross each input example
        activations = updateLoop[-1]

        Layer.__init__(self, [nNodes, nNodes], activations, dropout)
  
    def resetHiddenLayer(self):
        self.h = tf.Variable(tf.zeros([1, self.shape[0]]))


def GRU(Layer):
 
    def __init__( inLayer, nNodes, dropout=False):
        
        inLayer = inLayer
        
        # Weight matrices. x: input hidden, h: recurrent hidden, y: output 
        xW = tf.Variable(tf.truncated_normal([inLayer.shape[1], nNodes], stddev=0.1))
        hW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        # Update gate weights
        xUW = tf.Variable(tf.truncated_normal([inLayer.shape[1], nNodes], stddev=0.1))
        hUW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        # Reset gate weights
        xRW = tf.Variable(tf.truncated_normal([inLayer.shape[1], nNodes], stddev=0.1))
        hRW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))

        # Biases
        xB = tf.Variable(tf.zeros([nNodes]))
        hB = tf.Variable(tf.zeros([nNodes]))
                
        xUB = tf.Variable(tf.zeros([nNodes]))
        hUB = tf.Variable(tf.zeros([nNodes]))

        xRB = tf.Variable(tf.zeros([nNodes]))
        hRB = tf.Variable(tf.zeros([nNodes]))
        
        # Hidden state/memory
        self.h = tf.Variable(tf.zeros([1, nNodes]))

        # Create [nIn x nNode] dimensional representations of x input with trainable weights
        xTransform = tf.matmul(inLayer.activations, xW) + xB
        xUpdates = tf.matmul(inLayer.activations, xUW) + xUB
        xResets = tf.matmul(inLayer.activations, xRW) + xRB

        # Loop counter
        index = tf.constant(0, dtype=tf.int32) 
        loopParams = [index,\
                      xTransform,\
                      self.h, \
                      tf.zeros([tf.shape(inLayer.activations)[0], nNodes], dtype=tf.float32)]
                      
        # Each iteration performs one step of RNN update, produces series of hidden states
        def updateLoopBody(idx, x, h, activations):
            # Grab weighted representation of the current input
            x_t = tf.slice(x, [idx, 0], [1, -1])
            xU_t = tf.slice(xUpdates, [idx, 0], [1, -1])
            xR_t = tf.slice(xResets, [idx, 0], [1, -1])

            # Reset and update gates
            u = tf.nn.sigmoid((tf.matmul(xU_t, xUW) + xUB) + (tf.matmul(h, hUW) + hUB))
            r = tf.nn.sigmoid((tf.matmul(xR_t, xRW) + xRB) + (tf.matmul(h, hRW) + hRB))           

            # Compute new h value, 
            hCandidate = tf.tanh(x_t + tf.matmul(hW, tf.mul(r, h)))
            h = tf.tanh(tf.mul((1-u), hCandidate) + tf.mul(u, hCandidate) + hB) 

            # This is an awkward way of getting activations to have the same shape as the targets.
            def firstIteration(): return h
            def nextIterations(): return tf.concat(0, [activations, h])
            activations = tf.cond(tf.equal(0, idx), firstIteration, nextIterations)
    
            idx = tf.add(idx, 1)

            return idx, x, h, activations

        # The update loop runs for each example in the batch.
        condition = lambda idx, x, h, activations: tf.less(idx, tf.shape(x)[0])
        updateLoop = tf.while_loop(condition, updateLoopBody, loopParams)

        # A time series of the RNN's hidden state accross each input example
        activations = updateLoop[-1]

        Layer.__init__(self, [nNodes, nNodes], activations, dropout)
  
    def resetHiddenLayer(self):
        self.h = tf.Variable(tf.zeros([1, self.shape[0]]))

 
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
   
    def rnnLayer(self, nNodes, dropout=False):
        self.__addLayer__(RNN(self.outLayer, nNodes, dropout))
 
    def gruLayer(self, nNodes, dropout=False):
        self.__addLayer__(RNN(self.outLayer, nNodes, dropout))
    
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
        
    def getFeedDict(self, inputs, keepProb=1, extras={}):

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

    def resetRecurrentHiddens(self):
        for layer in self.layers:
            if type(layer) is RNN:
                layer.resetHiddenLayer()

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
