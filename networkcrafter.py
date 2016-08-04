import numpy as np
import tensorflow as tf

class Layer:

    def __init__(self, shape, weightedInputs, activationFunction=None, dropout=False):

        self.weightedInputs = weightedInputs

        # Apply activation function
        if activationFunction is not None:
            self.activations = activationFunction(weightedInputs)
        else:
            self.activations = weightedInputs
          
        if not isinstance(self.activations, tf.Tensor):
            raise TypeError("A layer's activations must be of type TensorFlow.Tensor. Got: " + \
                str(type(self.activations)))

        self.shape = shape

        self.applyDropout = dropout       
        if dropout is True:
            self.keepProb = tf.placeholder(tf.float32) 
            self.activations = tf.nn.dropout(self.activations, self.keepProb)


class InputLayer(Layer):
 
    def __init__(self, nFeatures, dropout=False, applyOneHot=False, dtype=tf.float32):
        shape = [None, nFeatures]

        if applyOneHot:
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None])
            activations = tf.one_hot(self.inputs, nFeatures, dtype=dtype)
        else:
            self.inputs = tf.placeholder(dtype, shape=shape)
            activations = self.inputs
       
        Layer.__init__(self, shape, activations, dropout=dropout)


class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, activationFunction, dropout=False):
        
        shape = [inLayer.shape[1], nNodes]
        
        xavierStddev = sqrt(3/(shape[0]+shape[1]))
        self.weights = tf.Variable(tf.random_normal(shape=shape, stddev=xavierStddev))
        self.biases = tf.Variable(tf.random_normal(shape=[nNodes], stddev=xavierStddev))

        weightedInput = tf.matmul(inLayer.activations, self.weights) + self.biases
         
        Layer.__init__(self, shape, weightedInput, activationFunction, dropout=dropout)


class ConvLayer(Layer):
    
    def __init__(self, inLayer, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):

        self.weights = tf.Variable(tf.truncated_normal(filterSize, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[filterSize[-1]]))

        conv = tf.nn.conv2d(inLayer.activations, self.weights, strides, padding='SAME') + \
            self.biases
     
        Layer.__init__(self, filterSize, conv, activationFunction, dropout=dropout)

        
class PoolLayer(Layer):

    def __init__(self, inLayer, poolSize, strides, dropout=False, poolType=tf.nn.max_pool):
        
        activations = poolType(inLayer.activations, ksize=poolSize, strides=strides, padding='SAME')

        Layer.__init__(self, poolSize, activations, dropout=dropout)


class ReshapeLayer(Layer):

    def __init__(self, inLayer, newShape, dropout=False):
        activations = tf.reshape(inLayer.activations, newShape, name='reshapeLayer');
        
        Layer.__init__(self, newShape, activations, dropout=dropout)


class RNN(Layer):

    def __init__(self, inLayer, nNodes, dropout=False):
        
        self.inLayer = inLayer
        
        # Inputs get re-represented in same dimensions as hidden state
        xTransform = FullConnectLayer(inLayer, nNodes, None).activations
 
        # Weights for recurrent connection
        self.hW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        self.hB = tf.Variable(tf.zeros([nNodes]))
        
        # Hidden state/memory
        self.h = tf.Variable(tf.zeros([nNodes]))

        # Function used by scan, applied to each input example
        def scanInputs(h,x):
            # Need to add a dim to h so we can apply matrix multiplication (recurrent update)
            h_row = tf.expand_dims(h,0)
            uH = tf.matmul(h_row, self.hW) + self.hB
            # Add the inputs, squeeze it back to rank 1. Will be packed into a matrix by scan().
            hPlusX = tf.squeeze(tf.tanh(x + uH))
            
            return hPlusX

        activations = tf.scan(scanInputs, xTransform, initializer=self.h)
 
        Layer.__init__(self, [nNodes, nNodes], activations, dropout=dropout)
  
    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([self.shape[0]])).eval(session=sess)


class GRU(Layer):
 
    def __init__(self, inLayer, nNodes, dropout=False):

        nTimeSteps = tf.shape(inLayer.activations)[0]
        
        # Rerepresent inputs with same Dims as hidden state. Also reset and input gates.
        xTransform = FullConnectLayer(inLayer, nNodes, None).activations
        xResets    = FullConnectLayer(inLayer, nNodes, None).activations
        xUpdates   = FullConnectLayer(inLayer, nNodes, None).activations

        # Recurrent weights, update and reset weights
        hW  = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        hUW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        hRW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))

        # Biases
        hB  = tf.Variable(tf.zeros([nNodes]))
        hUB = tf.Variable(tf.zeros([nNodes]))
        hRB = tf.Variable(tf.zeros([nNodes]))
        
        # Hidden state/memory
        self.h = tf.Variable(tf.zeros([1, nNodes]))

        # Loop counter
        index = tf.constant(0, dtype=tf.int32) 
        loopParams = [index,\
                      self.h,\
                      tf.TensorArray(dtype=tf.float32, size=nTimeSteps, dynamic_size=False)]
 
        # Each iteration performs one step of RNN update, produces series of hidden states
        def updateLoopBody(idx, h, hSequence):
            # Grab weighted representation of the current input
            x_t  = tf.slice(xTransform, [idx, 0], [1, -1])
            xU_t = tf.slice(xUpdates, [idx, 0], [1, -1])
            xR_t = tf.slice(xResets, [idx, 0], [1, -1])

            # Reset and update gates
            u = tf.nn.sigmoid(xU_t + tf.matmul(h, hUW) + hUB)
            r = tf.nn.sigmoid(xR_t + tf.matmul(h, hRW) + hRB)           

            # Compute new h value, 
            hCandidate = tf.tanh(x_t + tf.matmul(tf.mul(r, h), hW) + hB)
            h = tf.mul((1-u), hCandidate) + tf.mul(u, hCandidate)

            return idx+1, self.h.assign(h), hSequence.write(idx, h)

        # The update loop runs for each example in the batch.
        condition = lambda idx, h, activations: tf.less(idx, nTimeSteps)
        _, _, hStates = tf.while_loop(condition, updateLoopBody, loopParams)

        Layer.__init__(self, [nNodes, nNodes], hStates.concat(), dropout=dropout)
  
    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([1, self.shape[0]])).eval(session=sess)
 

class DynamicGRU(Layer):

    def __init__(self, inLayer, nNodes, nLayers=1, batchSize=1, dropout=False):

        # TensorFlow's build in GRU cell
        cell = tf.nn.rnn_cell.GRUCell(nNodes)

        # Can stack multiple layers
        assert nLayers > 0  
        if nLayers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell]*nLayers)
        self.nLayers = nLayers
        self.batchSize = batchSize
         
        # Assumption that data has rank-2 on the way in, reshape to get a batch of sequences
        sequence = tf.reshape(inLayer.activations, [batchSize, -1, inLayer.shape[-1]])

        self.h = tf.Variable(tf.zeros([batchSize, nNodes*nLayers]))

        outputs, state = tf.nn.dynamic_rnn(cell, sequence, initial_state=self.h, dtype=tf.float32)
        
        # Control depency forces the hidden state to persist
        with tf.control_dependencies([self.h.assign(state)]):
            # Squeeze the batches back together
            activations = tf.reshape(outputs, [-1, nNodes])

  
        Layer.__init__(self, [nNodes, nNodes], activations, dropout=dropout) 

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([self.batchSize, self.shape[0]*self.nLayers])).eval(session=sess)


class Network:

    def inputLayer(self, nFeatures, dropout=False, applyOneHot=False, dtype=tf.float32):
        self.inLayer = InputLayer(nFeatures, dropout, applyOneHot, dtype)
        self.layers = [self.inLayer]
        self.hiddens = []
        self.outLayer = self.inLayer
      
    def defineTargets(self, nNodes, applyOneHot, dtype=tf.float32):
        self.targets = InputLayer(nNodes, False, applyOneHot, dtype)
        self.targetVals = self.targets.activations

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
        self.__addLayer__(GRU(self.outLayer, nNodes, dropout))

    def dynamicGRU(self, nNodes, nLayers=1, batchSize=1, dropout=False):
        self.__addLayer__(DynamicGRU(self.outLayer, nNodes, nLayers, batchSize, dropout=dropout))

    def __addLayer__(self, layer):
        self.layers.append(layer)
        self.hiddens.append(layer)
        self.outLayer = layer

    def __init__(self):
        self.hiddens =[]

        
    def forward(self, sess, inputs, keepProb=1, batchSize=1):
        """Do a forward pass through the network and return the result.

        inputs -- The input data
        keepProb -- For layers with dropout, the probability of keeping each node

        """
        feedDict = self.getFeedDict(inputs, keepProb)

        output = self.outLayer.activations        
        if batchSize > 1: 
            output = tf.reshape(output, [batchSize, -1, self.outLayer.shape[-1]])
            

        return sess.run(output, feed_dict=feedDict)

        
    def getFeedDict(self, inputs, keepProb=1, extras={}, targets=None):

        """Create a feed dicionary for tensorflow. 

        inputs -- The input data to be slotted into the input layer activations
        keepProb -- For layers with dropout, the probability of keeping each node
        extras -- Any values to be added to the dictionary that are external to the network
        """
        # Define the feed_dict for a forward pass
        feedDict = {self.inLayer.inputs: inputs}
        # For each layer with dropout, add its keepProb to feed_dict
        possibleDropoutLayers = [self.inLayer]
        possibleDropoutLayers.extend(self.hiddens)
        for layer in possibleDropoutLayers:
            if layer.applyDropout:
                feedDict[layer.keepProb] = keepProb
        
        if targets is not None:
            feedDict[self.targets.inputs] = targets

        feedDict.update(extras) 

        return feedDict 

    def resetRecurrentHiddens(self, sess):
        for layer in self.layers:
            if isinstance(layer, RNN) or isinstance(layer, GRU) or isinstance(layer, DynamicGRU):
                layer.resetHiddenLayer(sess)
             

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
