import numpy as np
import tensorflow as tf

class Layer:

    def __init__(self, shape, activationFunction=None, dropout=False):

        self.activationFunction = activationFunction
        self.shape = shape
        self.keepProb = None
        self.applyDropout = dropout

    def buildGraph(self, weightedInputs):
        # Apply activation function
        self.weightedInputs = weightedInputs

        if self.activationFunction is not None:
            self.activations = self.activationFunction(weightedInputs)
        else:
            self.activations = weightedInputs
        if not isinstance(self.activations, tf.Tensor):
            raise TypeError("A layer's activations must be of type TensorFlow.Tensor. Got: " + \
                str(type(self.activations)))

        if self.applyDropout is True:
            self.keepProb = tf.placeholder(tf.float32)
            self.activations = tf.nn.dropout(self.activations, self.keepProb)


class InputLayer(Layer):

    def __init__(self, nFeatures, applyOneHot=False, dtype=tf.float32):
        shape = [None, nFeatures]

        if applyOneHot:
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None])
            self.activations = tf.one_hot(self.inputs, nFeatures, dtype=dtype)
        else:
            self.inputs = tf.placeholder(dtype=dtype, shape=shape)
            self.activations = self.inputs

        Layer.__init__(self, shape)

    def buildGraph(self):
        Layer.buildGraph(self, self.activations)


class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, activationFunction, dropout=False, wb=None):

        self.inLayer = inLayer
        shape = [inLayer.shape[-1], nNodes]

        # Weights and biases. If None were passed in, automatically intialize them.
        if wb is None:
            self.weights, self.biases = FullConnectLayer.xavierInit(shape)
        else:
            self.weights = wb[0]
            self.biases = wb[1]

        Layer.__init__(self, shape, activationFunction, dropout=dropout)

    @staticmethod
    def xavierInit(shape):
        xavierStddev = np.sqrt(3.0/(shape[0]+shape[1]))
        weights = tf.Variable(tf.random_normal(shape=shape, stddev=xavierStddev))
        biases = tf.Variable(tf.random_normal(shape=[shape[1]], stddev=xavierStddev))
        return [weights, biases]

    def buildGraph(self): 
        weightedInput = tf.matmul(self.inLayer.activations, self.weights) + self.biases
        Layer.buildGraph(self, weightedInput)


class ConvLayer(Layer):

    def __init__(self, inLayer, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):

        self.inLayer = inLayer
        self.strides = strides
        self.weights = tf.Variable(tf.truncated_normal(filterSize, stddev=0.1))
        self.biases = tf.Variable(tf.constant(0.1, shape=[filterSize[-1]]))
        Layer.__init__(self, filterSize, activationFunction, dropout=dropout)

    def buildGraph(self):        
        conv = tf.nn.conv2d(self.inLayer.activations, self.weights, self.strides,\
            padding='SAME') + self.biases
        Layer.buildGraph(self, conv)


class PoolLayer(Layer):

    def __init__(self, inLayer, poolSize, strides, dropout=False, poolType=tf.nn.max_pool):

        self.inLayer = inLayer
        self.poolSize = poolSize
        self.strides = strides
        self.poolType = poolType
        Layer.__init__(self, poolSize, dropout=dropout)

    def buildGraph(self):
        pool = self.poolType(self.inLayer.activations, ksize=self.poolSize, \
            strides=self.strides, padding='SAME')
        Layer.buildGraph(self, pool)


class ReshapeLayer(Layer):

    def __init__(self, inLayer, newShape, dropout=False):
        self.inLayer = inLayer
        Layer.__init__(self, newShape, dropout=dropout)

    def buildGraph(self):
        activations = tf.reshape(self.inLayer.activations, self.shape)
        Layer.buildGraph(self, activations)

class RNN(Layer):

    def __init__(self, inLayer, nNodes, dropout=False):

        self.inLayer = inLayer

        # Inputs get re-represented in same dimensions as hidden state
        self.xTransform = FullConnectLayer(inLayer, nNodes, None)

        # Weights for recurrent connection
        self.hW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        self.hB = tf.Variable(tf.zeros([nNodes]))

        # Hidden state/memory
        self.h = tf.Variable(tf.zeros([nNodes]))

        Layer.__init__(self, [nNodes, nNodes], dropout=dropout)

    def buildGraph(self):
        self.xTransform.buildGraph()    

        # Function used by scan, applied to each input example
        def scanInputs(h,x):
            # Need to add a dim to h so we can apply matrix multiplication (recurrent update)
            h_row = tf.expand_dims(h,0)
            uH = tf.matmul(h_row, self.hW) + self.hB
            # Add the inputs, squeeze it back to rank 1. Will be packed into a matrix by scan().
            hPlusX = tf.squeeze(tf.tanh(x + uH))

            return hPlusX

        activations = tf.scan(scanInputs, xTransform.activations, initializer=self.h)
        Layer.buildGraph(self, activations)

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([self.shape[0]])).eval(session=sess)


class GRU(Layer):

    def __init__(self, inLayer, nNodes, dropout=False):

        self.nTimeSteps = tf.shape(inLayer.activations)[0]

        # Rerepresent inputs with same Dims as hidden state. Also reset and input gates.
        self.xTransform = FullConnectLayer(inLayer, nNodes, None)
        self.xResets    = FullConnectLayer(inLayer, nNodes, None)
        self.xUpdates   = FullConnectLayer(inLayer, nNodes, None)

        # Recurrent weights, update and reset weights
        self.hW  = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        self.hUW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))
        self.hRW = tf.Variable(tf.truncated_normal([nNodes, nNodes], stddev=0.1))

        # Biases
        self.hB  = tf.Variable(tf.zeros([nNodes]))
        self.hUB = tf.Variable(tf.zeros([nNodes]))
        self.hRB = tf.Variable(tf.zeros([nNodes]))

        # Hidden state/memory
        self.h = tf.Variable(tf.zeros([1, nNodes]))

        Layer.__init__(self, [nNodes, nNodes], dropout=dropout)

    def buildGraph(self):

        self.xTransform.buildGraph()
        self.xResets.buildGraph()
        self.xUpdates.buildGraph()
        
        # Loop counter
        index = tf.constant(0, dtype=tf.int32)
        loopParams = [index,\
                      self.h,\
                      tf.TensorArray(dtype=tf.float32, size=self.nTimeSteps, dynamic_size=False)]

        # Each iteration performs one step of RNN update, produces series of hidden states
        def updateLoopBody(idx, h, hSequence):
            # Grab weighted representation of the current input
            x_t  = tf.slice(self.xTransform.activations, [idx, 0], [1, -1])
            xU_t = tf.slice(self.xUpdates.activations, [idx, 0], [1, -1])
            xR_t = tf.slice(self.xResets.activations, [idx, 0], [1, -1])

            # Reset and update gates
            u = tf.nn.sigmoid(xU_t + tf.matmul(h, self.hUW) + self.hUB)
            r = tf.nn.sigmoid(xR_t + tf.matmul(h, self.hRW) + self.hRB)

            # Compute new h value,
            hCandidate = tf.tanh(x_t + tf.matmul(tf.mul(r, h), self.hW) + self.hB)
            h = tf.mul((1-u), hCandidate) + tf.mul(u, hCandidate)

            return idx+1, self.h.assign(h), hSequence.write(idx, h)

        # The update loop runs for each example in the batch.
        condition = lambda idx, h, activations: tf.less(idx, self.nTimeSteps)
        _, _, hStates = tf.while_loop(condition, updateLoopBody, loopParams)
    
        Layer.buildGraph(self, hStates.concat())

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([1, self.shape[0]])).eval(session=sess)


class DynamicGRU(Layer):

    def __init__(self, inLayer, nNodes, nLayers=1, batchSize=1, dropout=False):
        self.inLayer = inLayer
        self.nNodes = nNodes
        self.nLayers = nLayers
        self.batchSize = batchSize
        
        # TensorFlow's build in GRU cell
        self.cell = tf.nn.rnn_cell.GRUCell(nNodes)
        # Can stack multiple layers
        assert nLayers > 0
        if nLayers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*nLayers)

        self.h = tf.Variable(tf.zeros([batchSize, nNodes*nLayers]))
        Layer.__init__(self, [nNodes, nNodes], dropout=dropout)

    def buildGraph(self):
        # Assumption that data has rank-2 on the way in, reshape to get a batch of sequences
        self.sequence = \
            tf.reshape(self.inLayer.activations, [self.batchSize, -1, self.inLayer.shape[-1]])
        # Create outputs and state graph
        outputs, state = \
            tf.nn.dynamic_rnn(self.cell, self.sequence, initial_state=self.h, dtype=tf.float32)
        # Control depency forces the hidden state to persist
        with tf.control_dependencies([self.h.assign(state)]):
            # Squeeze the batches back together
            activations = tf.reshape(outputs, [-1, self.nNodes])
        
        Layer.buildGraph(self, activations)

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([self.batchSize, self.shape[0]*self.nLayers])).eval(session=sess)


class Seq2SeqBasic(Layer):

    def __init__(self, encodeInLayer, decodeInLayer, nNodes, enSeqLength, deSeqLength, wb=None):


        nFeaturesEn = encodeInLayer.shape[-1]
        nFeaturesDe = decodeInLayer.shape[-1]

        self.encodeInputs = tf.reshape(encodeInLayer.activations, [-1, enSeqLength, nFeaturesEn])
        self.decodeInputs = tf.reshape(decodeInLayer.activations, [-1, deSeqLength, nFeaturesDe])

        self.encodeInputs = tf.unpack(self.encodeInputs, self.enSeqLength, axis=1)
        self.decodeInputs = tf.unpack(self.decodeInputs, self.deSeqLength, axis=1)

        # Passed to decoder, determines whether to pass in the decodeInputs or the prvious pred
        self.feedPrev = tf.Variable(tf.constant(False))

        self.cell = tf.nn.rnn_cell.GRUCell(nNodes)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*2)

        def loopFunction(prev, i):
            feedDecodeIn = lambda : self.decodeInputs[i]
            # Use output projection weights if provided
            if wb is not None:
                prev = tf.matmul(prev, wb[0]) + wb[1]
            feedPredict = lambda : tf.one_hot(tf.argmax(prev, 1), nFeaturesDe)

            return tf.cond(self.feedPrev, feedPredict, feedDecodeIn)
        self.loopFunction = loopFunction

        Layer.__init__(self, [nNodes], activations)

    def buildGraph(self):

        _, encodedState = tf.nn.rnn(self.cell, self.encodeInputs, dtype=tf.float32)
        outputs, state = tf.nn.seq2seq.rnn_decoder(self.decodeInputs, self.encodedState, \
            self.cell, loop_function=self.loopFunction)

        activations = tf.concat(1, outputs)
        activations = tf.reshape(activations, [-1, nNodes])
        Layer.buildGraph(self, activations)

    def setFeedPrevious(self, boolVal, sess):
        self.feedPrev.assign(boolVal).eval(session=sess)


class Network:

    def inputLayer(self, nFeatures, applyOneHot=False, dtype=tf.float32):
        self.inLayer = InputLayer(nFeatures, applyOneHot, dtype)
        self.layers = [self.inLayer]
        self.hiddens = []
        self.outLayer = self.inLayer

    def defineDecodeInLayer(self, nFeatures, applyOneHot=False, dtype=tf.float32):
        self.decodeInLayer = InputLayer(nFeatures, applyOneHot, dtype)

    def defineTargets(self, nNodes, applyOneHot, dtype=tf.float32):
        self.targets = InputLayer(nNodes, applyOneHot, dtype)
        self.targetVals = self.targets.activations

    def fullConnectLayer(self,  nNodes, activationFunction, dropout=False, wb=None):
        self.__addLayer__(FullConnectLayer(self.outLayer, nNodes, activationFunction, dropout, wb))

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

    def seq2SeqBasic(self, nNodes, enSeqLen, deSeqLen, wb):
        if self.decodeInLayer is None:
            raise StandardError("Must define a decodeInLayer for the Seq2Seq model.")
        self.__addLayer__(Seq2SeqBasic(self.outLayer, self.decodeInLayer, nNodes,\
            enSeqLen, deSeqLen, wb))

    def __addLayer__(self, layer):
        self.layers.append(layer)
        self.hiddens.append(layer)
        self.outLayer = layer

    def __init__(self):
        self.decodeInLayer = None
        self.hiddens =[]

    def buildGraph(self):
        for layer in self.layers:
            layer.buildGraph()
        self.outputs = self.outLayer.activations


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


    def getFeedDict(self, inputs, keepProb=1, extras={}, targets=None, decoderInputs=None):

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
        if decoderInputs is not None:
            feedDict[self.decodeInLayer.inputs] = decoderInputs

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
            dropouts = [False]*(len(sizes)-2)

        if len(dropouts) != len(sizes)-2:
            raise RuntimeError("Must specify whether to apply dropout to each layer" + \
                                " (excluding in/output layer). Use None for default.")

        # Never use dropout on the last layer
        dropouts.append(False)

        # Initialize superclass and add the layers
        Network.__init__(self)

        self.inputLayer(sizes[0])
        for nNodes,activation,dropout in zip(sizes[1:], activations, dropouts):
            self.fullConnectLayer(nNodes, activation, dropout)
