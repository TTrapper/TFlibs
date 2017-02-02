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

    def __init__(self, nFeatures, applyOneHot=False, dtype=tf.float32, inputTensor=None):
        shape = [None, nFeatures]

        Layer.__init__(self, shape)
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

    def __init__(self, numEmbeddings, embeddingDim):

        Layer.__init__(self, shape=[numEmbeddings, embeddingDim])

        self.embeddings = tf.get_variable("embeddings", self.shape)
        self.idPlaceholder = tf.placeholder(dtype=tf.int32, shape=[None], name="idPlaceholder")
        self.inputs = self.idPlaceholder

    def buildGraph(self):
        activations = tf.nn.embedding_lookup(self.embeddings, self.idPlaceholder)
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
        activations = tf.concat(1, [self.inLayer.activations, self.concatTensor])
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

        activations = tf.scan(scanInputs, self.xTransform.activations, initializer=self.h)
        Layer.buildGraph(self, activations)

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([self.shape[0]])).eval(session=sess)


class GRU(Layer):

    def __init__(self, inLayer, nNodes, dropout=False):
        self.inLayer = inLayer

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
        self.nTimeSteps = tf.shape(self.inLayer.activations)[0]

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

    def __init__(self, inLayer, nNodes, nLayers=1, batchSize=1, dropout=False, initialState=None,
        saveState=True, activationsAreFinalState=False):

        self.inLayer = inLayer
        self.nNodes = nNodes
        self.nLayers = nLayers
        self.batchSize = batchSize
        self.saveState = saveState
        self.activationsAreFinalState = activationsAreFinalState
        self.sequenceLengths = tf.placeholder(dtype=tf.int32, shape=[batchSize], name="DyGRUSeqLen")

        # TensorFlow's build in GRU cell
        self.cell = tf.nn.rnn_cell.GRUCell(nNodes)
        # Can stack multiple layers
        assert nLayers > 0
        if nLayers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*nLayers)

        if initialState is None:
            if nLayers > 1:
                self.h = tf.Variable([[0]*nNodes]*nLayers)
            else:
                self.h = tf.Variable([0]*nNodes)
        else:
            self.h = tf.Variable(initialState)

        Layer.__init__(self, [nNodes, nNodes*nLayers], dropout=dropout)

    def buildGraph(self):
        # Assumption that data has rank-2 on the way in, reshape to get a batch of sequences
        self.sequence = \
            tf.reshape(self.inLayer.activations, [self.batchSize, -1, self.inLayer.shape[-1]])

        # Create outputs and state graph
        outputs, self.state = tf.nn.dynamic_rnn(self.cell, self.sequence, \
            initial_state=self.h, sequence_length=self.sequenceLengths, dtype=tf.float32)
        self.outputs = outputs

        if self.saveState:
            # Control depency forces the hidden state to persist
            with tf.control_dependencies([self.h.assign(self.state)]):
                if self.activationsAreFinalState:
                    activations = self.state
                else:
                    # Squeeze the batches back together
                    activations = tf.reshape(outputs, [-1, self.nNodes])
        else:
            if self.activationsAreFinalState:
                activations = self.state
            else:
                activations = tf.reshape(outputs, [-1, self.nNodes])

        Layer.buildGraph(self, activations)

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([self.batchSize, self.shape[0]*self.nLayers])).eval(session=sess)


class BasicGRU(Layer):

    def __init__(self, inLayer, nNodes, nLayers, maxSeqLen, batchSize=1, dropout=False, initialState=None,
        saveState=True, activationsAreFinalState=False):

        self.inLayer = inLayer
        self.nNodes = nNodes
        self.nLayers = nLayers
        self.maxSeqLen = maxSeqLen
        self.batchSize = batchSize
        self.saveState = saveState
        self.activationsAreFinalState = activationsAreFinalState
        self.sequenceLengths = tf.placeholder(dtype=tf.int32, name="BasicGRUSeqLen")

        # Tensorflow's built in GRU cell
        self.cell = tf.nn.rnn_cell.GRUCell(nNodes)
        assert nLayers > 0
        if nLayers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*nLayers)

        if initialState is None:
            self.h = self.cell.zero_state(batchSize, tf.float32)
            if nLayers > 1:
                self.h = tuple([tf.Variable(state, trainable=False) for state in self.h])
            else:
                self.h = tf.Variable(self.h, trainable=False)
        else:
            if saveState is True:
                raise ValueError("saveState is not supported with external initialState.")
            self.h = initialState

        Layer.__init__(self, [nNodes, nNodes], dropout=dropout)

    def buildGraph(self):
        # Assumption that data has rank-2 on the way in, reshape to get a batch of sequences
        self.sequence = \
            tf.reshape(self.inLayer.activations, [-1, self.maxSeqLen, self.inLayer.shape[-1]])
        self.sequence = tf.unpack(self.sequence, self.maxSeqLen, axis=1)

        # Create outputs and state graph
        outputs, self.state = tf.nn.rnn(self.cell, self.sequence, \
            initial_state=self.h, sequence_length=self.sequenceLengths, dtype=tf.float32)
        self.outSequence = tf.concat(1, outputs)

        if self.saveState:
            # Control depency forces the hidden state to persist
            with tf.control_dependencies([self._assignInitialStateOp(self.state)]):
                activations = self._getActivations(self.outSequence)
        else:
            activations = self._getActivations(self.outSequence)

        Layer.buildGraph(self, activations)

    def resetHiddenLayer(self, sess, newState=None):
        if newState is None:
            newState = self.cell.zero_state(self.batchSize, tf.float32)
        sess.run(self._assignInitialStateOp(newState))

    def _assignInitialStateOp(self, newState):
        if self.nLayers > 1:
            """ Bit of a hack here, returning a useless identy op.
                The desired assign op is forced by control_dependencies. """
            with tf.control_dependencies([self.h[i].assign(state) for i, state in enumerate(newState)]):
                return tf.identity(1)
        else:
            return self.h.assign(newState)

    def _getActivations(self, outputs):
        if self.activationsAreFinalState:
            if self.nLayers > 1:
                return tf.identity(self.state[-1])
            else:
                return tf.identity(self.state)
        else:
            # Stack the time and batch dimensions
            return tf.reshape(outputs, [-1, self.nNodes])

class Seq2SeqBasic(Layer):

    def __init__(self, encodeInLayer, decodeInLayer, nNodes, enSeqLength, deSeqLength,
        readout=None, feedPrev=False):

        self.nNodes = nNodes
        nFeaturesEn = encodeInLayer.shape[-1]
        nFeaturesDe = decodeInLayer.shape[-1]

        self.enSequenceLengths = \
            tf.placeholder(dtype=tf.int32, name="Seq2SeqEnLengths")

        self.encodeInputs = tf.reshape(encodeInLayer.activations, [-1, enSeqLength, nFeaturesEn])
        self.decodeInputs = tf.reshape(decodeInLayer.activations, [-1, deSeqLength, nFeaturesDe])

        self.encodeInputs = tf.unpack(self.encodeInputs, enSeqLength, axis=1)
        self.decodeInputs = tf.unpack(self.decodeInputs, deSeqLength, axis=1)

        # Passed to decoder, determines whether to pass in the decodeInputs or the prvious pred
        self.feedPrev = tf.Variable(tf.constant(feedPrev))

        self.cell = tf.nn.rnn_cell.GRUCell(nNodes)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*2)

        self.readout = readout
        def loopFunction(prev, i):
            feedDecodeIn = lambda : self.decodeInputs[i]
            # Use readout graph provided to set the previous value to the network's output
            if self.readout is not None:
                self.readout.layers[1].inLayer.activations = prev
                self.readout.buildGraph()
                prev = self.readout.outputs
            feedPredict = lambda : tf.one_hot(tf.argmax(prev, 1), nFeaturesDe)
            return tf.cond(self.feedPrev, feedPredict, feedDecodeIn)
        self.loopFunction = loopFunction

        Layer.__init__(self, [nNodes])

    def buildGraph(self):

        _, self.encodedState = tf.nn.rnn(self.cell, self.encodeInputs, dtype=tf.float32,\
            sequence_length=self.enSequenceLengths)
        outputs, state = tf.nn.seq2seq.rnn_decoder(self.decodeInputs, self.encodedState, \
            self.cell, loop_function=self.loopFunction)

        activations = tf.concat(1, outputs)
        activations = tf.reshape(activations, [-1, self.nNodes])
        Layer.buildGraph(self, activations)

    def setFeedPrevious(self, boolVal, sess):
        self.feedPrev.assign(boolVal).eval(session=sess)


class Seq2SeqDynamic(Layer):

    def __init__(self, encodeInLayer, decodeInLayer, nNodes, batchSize):
        self.nNodes = nNodes
        with tf.variable_scope("rnnEncode"):
            encodeLayer = DynamicGRU(\
                encodeInLayer, nNodes, nLayers=1, batchSize=batchSize, saveState=False)
            self.encodeLayer = encodeLayer
            encodeLayer.buildGraph()
        with tf.variable_scope("rnnDecode"):
            self.decodeLayer = DynamicGRU(decodeInLayer, nNodes, nLayers=1,\
                 batchSize=batchSize, initialState=encodeLayer.state, saveState=False)
            self.decodeLayer.buildGraph()

        Layer.__init__(self, [nNodes])

    def buildGraph(self):
        activations = self.decodeLayer.activations
        Layer.buildGraph(self, activations)


class Network:

    def inputLayer(self, nFeatures, applyOneHot=False, dtype=tf.float32, inputTensor=None):
        self.__addLayerWithScope__(InputLayer, nFeatures, applyOneHot, dtype, inputTensor)

    def embeddingLayer(self, numEmbeddings, embeddingDim):
        self.__addLayerWithScope__(EmbeddingLayer, numEmbeddings, embeddingDim)

    def defineDecodeInLayer(self, nFeatures, applyOneHot=False, dtype=tf.float32):
        self.decodeInLayer = InputLayer(nFeatures, applyOneHot, dtype)

    def defineTargets(self, nNodes, applyOneHot, dtype=tf.float32):
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

    def convLayer(self, activationFunction, filterSize, strides=[1,1,1,1], dropout=False):
        self.__addLayerWithScope__(
            ConvLayer, self.outLayer, activationFunction, filterSize, strides, dropout)

    def poolLayer(self,  poolSize, strides=None, dropout=False, poolType=tf.nn.max_pool):
        if strides is None:
            strides = poolSize
        self.__addLayerWithScope__(PoolLayer, self.outLayer, poolSize, strides, dropout, poolType)

    def reshapeLayer(self, newShape, dropout=False):
        self.__addLayerWithScope__(ReshapeLayer, self.outLayer, newShape, dropout)

    def rnnLayer(self, nNodes, dropout=False):
        self.__addLayerWithScope__(RNN, self.outLayer, nNodes, dropout)

    def gruLayer(self, nNodes, dropout=False):
        self.__addLayerWithScope__(GRU, self.outLayer, nNodes, dropout)

    def basicGRU(self, nNodes, nLayers=1, maxSeqLen=10, batchSize=1, dropout=False, initialState=None,
        saveState=True, activationsAreFinalState=False):

        self.__addLayerWithScope__(
            BasicGRU, self.outLayer, nNodes, nLayers, maxSeqLen, batchSize=batchSize,
            dropout=dropout, initialState=initialState, saveState=saveState,
            activationsAreFinalState=activationsAreFinalState)

    def dynamicGRU(self, nNodes, nLayers=1, batchSize=1, dropout=False, initialState=None,
        saveState=True, activationsAreFinalState=False):

        self.__addLayerWithScope__(DynamicGRU,
            self.outLayer, nNodes, nLayers, batchSize, dropout=dropout, initialState=initialState,
            saveState=saveState, activationsAreFinalState=activationsAreFinalState)

    def seq2SeqBasic(self, nNodes, enSeqLen, deSeqLen, wb, feedPrev=False):
        if self.decodeInLayer is None:
            raise StandardError("Must define a decodeInLayer for the Seq2Seq model.")
        self.__addLayerWithScope__(Seq2SeqBasic, self.outLayer, self.decodeInLayer, nNodes,\
            enSeqLen, deSeqLen, wb, feedPrev)

    def seq2SeqDynamic(self, nNodes, batchSize):
        if self.decodeInLayer is None:
            raise StandardError("Must define a decodeInLayer for the Seq2Seq model.")
        self.__addLayerWithScope__(
            Seq2SeqDynamic, self.outLayer, self.decodeInLayer, nNodes, batchSize)

    def __addLayerWithScope__(self, layerClass, *args, **kwargs):

        with tf.variable_scope(self.scope):
           layerScopeName = "layer" + str(len(self.layers)) + "_" + layerClass.__name__
           with tf.variable_scope(layerScopeName) as scope:
               self.__addLayer__(layerClass(*args, **kwargs))

    def __addLayer__(self, layer):
        if (len(self.layers) == 0):
            self.inLayer = layer
        self.layers.append(layer)
        self.outLayer = layer

    def __init__(self, scopeName="net", reuseVariables=False,
        initializer=tf.contrib.layers.xavier_initializer()):

        with tf.variable_scope(scopeName, reuse=reuseVariables, initializer=initializer) as scope:
            self.scope=scope
        self.decodeInLayer = None
        self.layers = []

    def buildGraph(self):
        for layer in self.layers:
            layer.buildGraph()
        self.outputs = self.outLayer.activations

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
            feedDict[self.inLayer.inputs] = inputs

        # Add keepProb to layers with dropout. Add sequence info to recurrent layers.
        for layer in self.layers:
            if layer.applyDropout:
                feedDict[layer.keepProb] = keepProb
            if isinstance(layer, Seq2SeqBasic):
                feedDict[layer.enSequenceLengths] = sequenceLengths
            if isinstance(layer, DynamicGRU) or isinstance(layer, BasicGRU):
                feedDict[layer.sequenceLengths] = sequenceLengths
            if isinstance(layer, Seq2SeqDynamic):
                feedDict[layer.encodeLayer.sequenceLengths] = sequenceLengths
                if decoderSequenceLengths is not None:
                    feedDict[layer.decodeLayer.sequenceLengths] = decoderSequenceLengths

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
