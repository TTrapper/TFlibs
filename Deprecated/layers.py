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
            hCandidate = tf.tanh(x_t + tf.matmul(tf.multiply(r, h), self.hW) + self.hB)
            h = tf.multiply((1-u), hCandidate) + tf.multiply(u, hCandidate)

            return idx+1, self.h.assign(h), hSequence.write(idx, h)

        # The update loop runs for each example in the batch.
        condition = lambda idx, h, activations: tf.less(idx, self.nTimeSteps)
        _, _, hStates = tf.while_loop(condition, updateLoopBody, loopParams)

        Layer.buildGraph(self, hStates.concat())

    def resetHiddenLayer(self, sess):
        self.h.assign(tf.zeros([1, self.shape[0]])).eval(session=sess)


class Seq2SeqBasic(Layer):

    def __init__(self, encodeInLayer, decodeInLayer, nNodes, enSeqLength, deSeqLength,
        readout=None, feedPrev=False):

        self.nNodes = nNodes
        nFeaturesEn = encodeInLayer.shape[-1]
        nFeaturesDe = decodeInLayer.shape[-1]
        self.nFeaturesEn = nFeaturesEn
        self.nFeaturesDe = nFeaturesDe

        self.enSequenceLengths = \
            tf.placeholder(dtype=tf.int32, name="Seq2SeqEnLengths")

        self.encodeInputs = tf.reshape(encodeInLayer.activations, [-1, enSeqLength, nFeaturesEn])
        self.decodeInputs = tf.reshape(decodeInLayer.activations, [-1, deSeqLength, nFeaturesDe])

        self.encodeInputs = tf.unstack(self.encodeInputs, enSeqLength, axis=1)
        self.decodeInputs = tf.unstack(self.decodeInputs, deSeqLength, axis=1)

        # Passed to decoder, determines whether to pass in the decodeInputs or the prvious pred
        self.feedPrev = tf.Variable(tf.constant(feedPrev))

        self.enCell = tf.contrib.rnn.GRUCell(nNodes)
        self.enCell = tf.contrib.rnn.MultiRNNCell([self.enCell]*2)
        self.deCell = tf.contrib.rnn.GRUCell(nNodes)
        self.deCell = tf.contrib.rnn.MultiRNNCell([self.deCell]*2)

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

        self.encodeInputs = tf.reshape(self.encodeInputs, [1, -1, self.nFeaturesEn])
        _, self.encodedState = tf.nn.dynamic_rnn(self.enCell, self.encodeInputs,\
            sequence_length=self.enSequenceLengths, dtype=tf.float32)
        outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(self.decodeInputs,
            initial_state=self.encodedState, cell=self.deCell, loop_function=self.loopFunction)

        activations = tf.concat(axis=1, values=outputs)
        activations = tf.reshape(activations, [-1, self.nNodes])
        Layer.buildGraph(self, activations)

    def setFeedPrevious(self, boolVal, sess):
        self.feedPrev.assign(boolVal).eval(session=sess)
