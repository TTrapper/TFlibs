import numpy as np
import tensorflow as tf

class Layer(object):

    def __init__(self, shape, activationFunction=None, dropout=False):

        self.activationFunction = activationFunction
        self.shape = shape
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
            self.keepProb = tf.placeholder(tf.float32, name="keepProb")
            self.activations = tf.nn.dropout(self.activations, self.keepProb)

class InputLayer(Layer):

    def __init__(self,
        nFeatures, applyOneHot=False, dtype=tf.float32, inputTensor=None, dropout=False):

        shape = [None, nFeatures]

        Layer.__init__(self, shape, dropout=dropout)
        if applyOneHot:
            if inputTensor is not None:
                self.inputs = inputTensor
            else:
                self.inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='InputLayer')
            self.activations = tf.one_hot(self.inputs, nFeatures, dtype=dtype)
        else:
            if inputTensor is not None:
                self.inputs = inputTensor
            else:
                self.inputs = tf.placeholder(dtype=dtype, shape=shape, name='InputLayer')
            self.activations = self.inputs
            Layer.buildGraph(self, self.activations)

    def buildGraph(self):
        pass


class EmbeddingLayer(Layer):

    def __init__(self,
        numEmbeddings, embeddingDim, lookupTensor=None, trainable=True, dropout=False):

        Layer.__init__(self, shape=[numEmbeddings, embeddingDim], dropout=dropout)

        self.embeddings = tf.get_variable("embeddings", self.shape, trainable=trainable)
        if lookupTensor is not None:
            self.inputs = lookupTensor
        else:
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None], name="idPlaceholder")

    def buildGraph(self):
        activations = tf.nn.embedding_lookup(self.embeddings, self.inputs)
        Layer.buildGraph(self, activations)


class FullConnectLayer(Layer):

    def __init__(self, inLayer, nNodes, activationFunction, dropout=False, addBias=True,
        wTranspose=False):

        self.inLayer = inLayer
        self.nNodes = nNodes
        self.addBias = addBias
        self.wTranspose = wTranspose
        shape = [inLayer.shape[-1], nNodes] if not self.wTranspose else [nNodes, inLayer.shape[-1]]

        self.weights, self.biases = self.initWeights(shape)

        Layer.__init__(self, shape, activationFunction, dropout=dropout)

    def initWeights(self, shape):
        weights = tf.get_variable("weights", shape)
        biases = False
        if self.addBias:
            biases = tf.get_variable("biases", [self.nNodes])
        return [weights, biases]

    def buildGraph(self):
        weightedInput = tf.matmul(
            self.inLayer.activations, self.weights, transpose_b=self.wTranspose)
        weightedInput = weightedInput + self.biases if self.addBias else weightedInput
        Layer.buildGraph(self, weightedInput)

class ConcatLayer(Layer):

    def __init__(self, inLayer, concatTensor, concatTensorLen, dropout=False):
        self.inLayer = inLayer
        self.concatTensor = concatTensor
        shape = [inLayer.shape[-1] + concatTensorLen]
        Layer.__init__(self, shape, dropout=dropout)

    def buildGraph(self):
        activations = tf.concat(axis=1, values=[self.inLayer.activations, self.concatTensor])
        Layer.buildGraph(self, activations)

class AdditionLayer(Layer):

    def __init__(self, inLayer, addTensor, dropout=False):
        self.inLayer = inLayer
        self.addTensor = addTensor
        shape = [inLayer.shape[-1]]
        Layer.__init__(self, shape, dropout=dropout)

    def buildGraph(self):
        activations = tf.add(self.inLayer.activations, self.addTensor)
        Layer.buildGraph(self, activations)

class ReduceOpLayer(Layer):

    def __init__(self, inLayer, axis=0, reduceOp=tf.reduce_mean, dropout=False):
        self.inLayer = inLayer
        self.reduceOp = reduceOp
        self.axis = axis

        Layer.__init__(self, [inLayer.shape[-1]], dropout=dropout)

    def buildGraph(self):
        activations = self.reduceOp(self.inLayer.activations, axis=self.axis)
        Layer.buildGraph(self, activations)

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
        activations = tf.reshape(self.inLayer.activations, self.shape, name="reshapeLayer")
        Layer.buildGraph(self, activations)


class RNN(Layer):

    def __init__(self, inLayer, cell, nNodes, nLayers, maxSeqLen, sequenceLengths=None, batchSize=1,
        dropout=False, initialState=None, saveState=True, activationsAreFinalState=False):

        self.inLayer = inLayer
        self.cell = cell
        self.nNodes = nNodes
        self.nLayers = nLayers
        self.maxSeqLen = maxSeqLen
        self.batchSize = batchSize
        self.dropout = dropout
        self.saveState = saveState
        self.activationsAreFinalState = activationsAreFinalState
        if sequenceLengths is None:
            self.sequenceLengths = tf.placeholder(dtype=tf.int32, name="BasicGRUSeqLen")
        else:
            self.sequenceLengths = sequenceLengths

        if dropout:
            self.keepProb = tf.placeholder(tf.float32, name="keepProb")
            self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.keepProb)

        assert nLayers > 0
        if nLayers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell]*nLayers)
        if initialState is None:
            self.zeroState = self.cell.zero_state(batchSize, tf.float32)
            if nLayers > 1:
                self.initialState = tuple(
                    [tf.Variable(state, trainable=False) for state in self.zeroState])
            else:
                self.initialState = tf.Variable(self.zeroState, trainable=False)
        else:
            if saveState is True:
                raise ValueError("saveState is not supported with external initialState.")
            self.initialState = initialState

        Layer.__init__(self, [nNodes, nNodes], dropout=False)

    def buildGraph(self):

        # Assumption that data has rank-2 on the way in, reshape to get a batch of sequences
        self.sequence = \
            tf.reshape(self.inLayer.activations, [self.batchSize, -1, self.inLayer.shape[-1]])

        # Create outputs and state graph
        outputs, self.finalState = tf.nn.dynamic_rnn(self.cell, self.sequence, \
            initial_state=self.initialState, sequence_length=self.sequenceLengths, dtype=tf.float32)
        self.outSequence = tf.concat(axis=1, values=outputs)

        if self.saveState:
            # Control depency forces the hidden state to persist
            with tf.control_dependencies([self._assignInitialStateOp(self.finalState)]):
                activations = self._getActivations(self.outSequence)
        else:
            activations = self._getActivations(self.outSequence)

        # Add an ops to graph for resetting the initial state to zero or some value
        self.setZeroState = self._assignInitialStateOp(self.zeroState)
        if self.nLayers == 1:
            self.newState = tf.placeholder(tf.float32)
        else:
            self.newState = []
            [self.newState.append(tf.placeholder(tf.float32)) for i in range(self.nLayers)]
        self.setNewState = self._assignInitialStateOp(self.newState)

        Layer.buildGraph(self, activations)
        # applyDropout was false during Layer.buildGraph so that keepProb was not overwritten
        self.applyDropout = self.dropout

    def resetHiddenLayer(self, sess, newState=None):
        if newState is None:
            sess.run(self.setZeroState)
        else:
            if self.nLayers == 1:
                feedDict = {self.newState:newState}
            else:
                feedDict = {}
                for i in range(self.nLayers):
                    feedDict.update({self.newState[i]: newState[i]})
            sess.run(self.setNewState, feed_dict=feedDict)


    def _assignInitialStateOp(self, newState):
        if self.nLayers > 1:
            """ Bit of a hack here, returning a useless identy op.
                The desired assign op is forced by control_dependencies. """
            with tf.control_dependencies(
                [self.initialState[i].assign(state) for i, state in enumerate(newState)]):
                return tf.identity(1)
        else:
            return self.initialState.assign(newState)

    def _getActivations(self, outputs):
        if self.activationsAreFinalState:
            if self.nLayers > 1:
                return tf.identity(self.finalState[-1])
            else:
                return tf.identity(self.finalState)
        else:
            # Stack the time and batch dimensions
            return tf.reshape(outputs, [-1, self.nNodes])


class BasicGRU(RNN):

    def __init__(self, inLayer, nNodes, nLayers, maxSeqLen, sequenceLengths=None, batchSize=1,
        dropout=False, initialState=None, saveState=True, activationsAreFinalState=False):

        # Tensorflow's built in GRU cell
        self.cell = tf.contrib.rnn.GRUCell(nNodes)

        RNN.__init__(self, inLayer                  = inLayer,
                           cell                     = self.cell,
                           nNodes                   = nNodes,
                           nLayers                  = nLayers,
                           maxSeqLen                = maxSeqLen,
                           sequenceLengths          = sequenceLengths,
                           batchSize                = batchSize,
                           dropout                  = dropout,
                           initialState             = initialState,
                           saveState                = saveState,
                           activationsAreFinalState = activationsAreFinalState)

class Network:

    def inputLayer(self,
        nFeatures, applyOneHot=False, dtype=tf.float32, inputTensor=None, dropout=False):
        self.__addLayerWithScope__(InputLayer, nFeatures, applyOneHot, dtype, inputTensor, dropout)

    def embeddingLayer(self,
        numEmbeddings, embeddingDim, lookupTensor=None, trainable=True, dropout=False):
        self.__addLayerWithScope__(EmbeddingLayer, numEmbeddings, embeddingDim, lookupTensor,
            trainable=trainable, dropout=dropout)

    def defineDecodeInLayer(self, nFeatures, applyOneHot=False, dtype=tf.float32):
        self.decodeInLayer = InputLayer(nFeatures, applyOneHot, dtype)

    def defineTargets(self, nNodes, applyOneHot, dtype=tf.float32):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("targets"):
                self.targets = InputLayer(nNodes, applyOneHot, dtype)
                self.targetVals = self.targets.activations

    def fullConnectLayer(self,  nNodes, activationFunction, dropout=False, addBias=True,
        wTranspose=False):
        self.__addLayerWithScope__(FullConnectLayer, self.outLayer, nNodes, activationFunction,
                dropout, addBias, wTranspose)

    def concatLayer(self, concatTensor, concatTensorLen, dropout=False):
        self.__addLayerWithScope__(
            ConcatLayer, self.outLayer, concatTensor, concatTensorLen, dropout)

    def additionLayer(self, addTensor, dropout=False):
        self.__addLayerWithScope__(AdditionLayer, self.outLayer, addTensor, dropout)

    def reduceOpLayer(self, axis=0, reduceOp=tf.reduce_mean, dropout=False):
        self.__addLayerWithScope__(ReduceOpLayer, self.outLayer, axis, reduceOp, dropout)

    def convLayer(self, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):
        self.__addLayerWithScope__(
            ConvLayer, self.outLayer, activationFunction, filterSize, strides, dropout)

    def poolLayer(self,  poolSize, strides=None, dropout=False, poolType=tf.nn.max_pool):
        if strides is None:
            strides = poolSize
        self.__addLayerWithScope__(PoolLayer, self.outLayer, poolSize, strides, dropout, poolType)

    def reshapeLayer(self, newShape, dropout=False):
        self.__addLayerWithScope__(ReshapeLayer, self.outLayer, newShape, dropout)

    def basicGRU(self,
        nNodes, nLayers=1, maxSeqLen=10, sequenceLengths=None, batchSize=1, dropout=False,
        initialState=None, saveState=True, activationsAreFinalState=False):

        self.__addLayerWithScope__(
            BasicGRU, self.outLayer, nNodes, nLayers, maxSeqLen, sequenceLengths=sequenceLengths,
            batchSize=batchSize, dropout=dropout, initialState=initialState, saveState=saveState,
            activationsAreFinalState=activationsAreFinalState)

    def __addLayerWithScope__(self, layerClass, *args, **kwargs):
        with tf.variable_scope(self.scope):
            layerScopeName = "layer" + str(len(self.layers)) + "_" + layerClass.__name__
            with tf.variable_scope(layerScopeName):
                self.__addLayer__(layerClass(*args, **kwargs))

    def __addLayer__(self, layer):
        if (len(self.layers) == 0):
            self.inLayer = layer
        self.layers.append(layer)
        self.outLayer = layer

    def __init__(self, scopeName="net", reuseVariables=False,
        initializer=tf.contrib.layers.xavier_initializer()):
        self.scopeName = scopeName
        self.reuseVariables = reuseVariables

        with tf.variable_scope(scopeName, reuse=reuseVariables, initializer=initializer) as scope:
            self.scope = scope

        self.decodeInLayer = None
        self.layers = []

    def buildGraph(self):
        for i, layer in enumerate(self.layers):
            with tf.variable_scope(self.scope):
                layerScopeName = "layer" + str(i) + "_" + type(layer).__name__
                with tf.variable_scope(layerScopeName):
                        layer.buildGraph()
        self.outputs = self.outLayer.activations
        self.inputs = None if not hasattr(self.inLayer, 'inputs') else self.inLayer.inputs


    def setInputs(self, inputTensor):
        self.inLayer.activations = inputTensor

    def graftOn(self, net):
        net.layers[1].inLayer = self.outLayer
        for i, layer in enumerate(net.layers):
            if i > 0:
                self.__addLayer__(layer)

    def forward(self, sess, inputs, keepProb=1, batchSize=1, decoderInputs=None,
        sequenceLengths=None, decoderSequenceLengths=None):
        """Do a forward pass through the network and return the result.

        inputs -- The input data
        keepProb -- For layers with dropout, the probability of keeping each node

        """
        feedDict = self.getFeedDict(inputs, keepProb, decoderInputs=decoderInputs,
            sequenceLengths=sequenceLengths, decoderSequenceLengths=decoderSequenceLengths)

        output = self.outLayer.activations
        if batchSize > 1:
            output = tf.reshape(output, [batchSize, -1, self.outLayer.shape[-1]])


        return sess.run(output, feed_dict=feedDict)


    def getFeedDict(self, inputs, keepProb=1, extras={}, targets=None, decoderInputs=None,
        sequenceLengths=None, decoderSequenceLengths=None):

        """Create a feed dicionary for tensorflow.

        inputs -- The input data to be slotted into the input layer activations
        keepProb -- For layers with dropout, the probability of keeping each node
        extras -- Any values to be added to the dictionary that are external to the network
        """
        feedDict = {}
        # Define the feed_dict for a forward pass
        if inputs is not None:
            feedDict[self.inputs] = inputs

        # Add keepProb to layers with dropout. Add sequence info to recurrent layers.
        for layer in self.layers:
            if layer.applyDropout:
                feedDict[layer.keepProb] = keepProb
            if isinstance(layer, BasicGRU):
                if sequenceLengths is not None:
                    feedDict[layer.sequenceLengths] = sequenceLengths

        if targets is not None:
            feedDict[self.targets.inputs] = targets
        if decoderInputs is not None:
            feedDict[self.decodeInLayer.inputs] = decoderInputs

        feedDict.update(extras)
        return feedDict

    def resetRecurrentHiddens(self, sess):
        for layer in self.layers:
            if isinstance(layer, BasicGRU):
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
