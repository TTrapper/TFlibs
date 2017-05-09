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

    def __init__(self, inLayer, nNodes, nLayers, maxSeqLen, sequenceLengths=None, batchSize=1,
        keepProb=1.0, saveState=True, activationsAreFinalState=False):

        assert 0.0 <= keepProb and keepProb <= 1.0
        assert nLayers > 0
        assert nNodes > 0
        assert maxSeqLen > 0

        self.cells = None
        self.initialStates = None
        self.inLayer = inLayer
        self.nNodes = nNodes
        self.nLayers = nLayers
        self.maxSeqLen = maxSeqLen
        self.batchSize = batchSize
        self.keepProb = keepProb
        self.saveState = saveState
        self.activationsAreFinalState = activationsAreFinalState
        if sequenceLengths is None:
            self.sequenceLengths = tf.placeholder(dtype=tf.int32, name="BasicGRUSeqLen")
        else:
            self.sequenceLengths = sequenceLengths

        Layer.__init__(self, [nNodes, nNodes], dropout=False)

    def _createCells(self):
        # A list of cells, each with possibly multiple layers, each possibly with dropout
        cells = []
        for _ in range(self._numCells):
            cells.append([self._maybeDropoutCell() for layer in range(self.nLayers)])
        self.cells = self._consolodateLayers(cells)

    def _consolodateLayers(self, cellsAsLayerLists):
        if self.nLayers > 1:
            cells = [tf.contrib.rnn.MultiRNNCell(layers) for layers in cellsAsLayerLists]
        else:
            cells = [cell[0] for cell in cellsAsLayerLists]
        return cells

    def _maybeDropoutCell(self):
        keepProb = self.keepProb
        cell = self._createCell()
        return cell if keepProb == 1.0 else tf.contrib.rnn.DropoutWrapper(cell, keepProb)

    # Initial states stored as list of tuples:
    #   [cell0:(layer0, layer1, ...), cell1:(layer0, layer1, ...), ...]
    def _createInitialStates(self, cells):
        self.zeroStates = []
        for cell in cells:
            layerStates = cell.zero_state(self.batchSize, dtype=tf.float32)
            layerStates = layerStates if type(layerStates) is tuple else tuple([layerStates])
            self.zeroStates.append(layerStates)
        self.initialStates = []
        for cellStates in self.zeroStates:
            layerStates = [tf.Variable(layerState, trainable=False) for layerState in cellStates]
            self.initialStates.append(tuple(layerStates))

    # Sets up ops for resetting the initial state to zero or placeholder values
    def _createInitialStateResetOps(self, initialStates):
        self._setZeroState = self._assignInitialStateOp(self.zeroStates)
        newStates = []
        for cellState in initialStates:
             newStates.append([tf.placeholder(tf.float32) for layer in cellState])
        self._newStatePlaces = newStates
        self._setNewStates = self._assignInitialStateOp(self._newStatePlaces)

    # Returns an op assigning new tensors to the intial state variables
    # Takes a list of tuples, each tuple representing the layers for each cell
    def _assignInitialStateOp(self, newStates):
        oldStates = self.initialStates
        assert len(newStates) == len(oldStates)

        assignOps = []
        for oldCell, newCell in zip(oldStates, newStates):
            assert len(oldCell) == len(newCell)
            ops = [oldLayer.assign(newLayer) for oldLayer, newLayer in zip(oldCell, newCell)]
            assignOps.extend(ops)

        return tf.group(*assignOps, name='assignNewStates')

    # Outputs is an iterable, one tensor per cell
    def _getActivations(self, outputs, finalStates):
        lastLayerIdx = -1
        if self.activationsAreFinalState:
            states = [cellStates[lastLayerIdx] for cellStates in finalStates]
            return tf.concat(states, axis=1)
        else:
            # Concatenate the cells' outputs together
            outputs = tf.concat(outputs, axis=2)
            # Stack the time and batch dimensions
            return tf.reshape(outputs, [-1, self.nNodes])

    def buildGraph(self):
        self._createCells()
        self._createInitialStates(self.cells)
        self._createInitialStateResetOps(self.initialStates)

        # Assumption that data has rank-2 on the way in, reshape to get a batch of sequences
        sequence = \
            tf.reshape(self.inLayer.activations, [self.batchSize, -1, self.inLayer.shape[-1]])
        outSequence, finalStates = self._unroller(sequence)

        if self.saveState:
            # Control depency forces the hidden state to persist
            with tf.control_dependencies([self._assignInitialStateOp(finalStates)]):
                activations = self._getActivations(outSequence, finalStates)
        else:
            activations = self._getActivations(outSequence, finalStates)


        self.outSequence = outSequence
        self.finalStates = finalStates
        Layer.buildGraph(self, activations)

    # Turn list of lists or list of tuples into a flat list
    def _flattenStates(self, states):
        flatStates = []
        for cell in states:
            flatStates.extend([layerState for layerState in cell])
        return flatStates

    def resetHiddenLayer(self, sess, newStates=None):
        if newStates is None:
            sess.run(self._setZeroState)
        else:
            newStates = self._flattenStates(newStates)
            newStatePlaces = self._flattenStates(self._newStatePlaces)
            feedDict = {}
            for i in range(self._numCells * self.nLayers):
                feedDict.update({newStatePlaces[i]: newStates[i]})
            sess.run(self._setNewStates, feed_dict=feedDict)

class BasicGRU(RNN):

    def __init__(self, inLayer, nNodes, nLayers, maxSeqLen, sequenceLengths=None, batchSize=1,
        keepProb=1.0, saveState=True, activationsAreFinalState=False):

        self._numCells = 1

        RNN.__init__(self, inLayer                  = inLayer,
                           nNodes                   = nNodes,
                           nLayers                  = nLayers,
                           maxSeqLen                = maxSeqLen,
                           sequenceLengths          = sequenceLengths,
                           batchSize                = batchSize,
                           keepProb                 = keepProb,
                           saveState                = saveState,
                           activationsAreFinalState = activationsAreFinalState)

    def _createCell(self):
        return tf.contrib.rnn.GRUCell(self.nNodes)

    def _unroller(self, sequences):
        cell = self.cells[0]

        initialState = self.initialStates[0] if self.nLayers > 1 else self.initialStates[0][0]
        sequenceLengths = self.sequenceLengths
        # Create outputs and state graph
        outputs, finalState = tf.nn.dynamic_rnn(cell, sequences,
            initial_state=initialState, sequence_length=sequenceLengths, dtype=tf.float32)
        # RNN expects a list of outputs for each cell, and a list of tuples for the states
        finalState = [finalState] if type(finalState) is tuple else [tuple([finalState])]

        return [outputs], finalState

class BidirectionalGRU(RNN):

    def __init__(self, inLayer, nNodes, nLayers, maxSeqLen, sequenceLengths=None, batchSize=1,
        dropout=False, initialState=None, saveState=True, activationsAreFinalState=False):



        pass

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

    def basicGRU(self, nNodes, nLayers=1, maxSeqLen=10, sequenceLengths=None, batchSize=1,
        keepProb=1.0, saveState=True, activationsAreFinalState=False):

        self.__addLayerWithScope__(
            BasicGRU, self.outLayer, nNodes, nLayers, maxSeqLen, sequenceLengths=sequenceLengths,
            batchSize=batchSize, keepProb=keepProb, saveState=saveState,
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
