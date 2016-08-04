import numpy as np
import networkcrafter as nc
import tensorflow as tf
import PythonTools.batchingTools as bt


BATCH_SIZE = 2

enData = np.array([[0], [1], [0]])
deData = np.array([[1], [0], [0], [1]])

enSeqLength = enData.shape[0]
deSeqLength = deData.shape[0]

enData = np.concatenate([enData]*BATCH_SIZE, axis=0)
deData = np.concatenate([deData]*BATCH_SIZE, axis=0)

print enData
targs = deData
targets = np.zeros([deData.shape[0], 2])

for i, t in enumerate(targs):
    targets[i,t] = 1

print targets

enLayer = nc.InputLayer(nFeatures=1)
deLayer = nc.InputLayer(nFeatures=1)

seq2seq = nc.Seq2SeqBasic(enLayer, deLayer, 6, enSeqLength, deSeqLength)

readout = nc.FullConnectLayer(seq2seq, nNodes=2, activationFunction=tf.nn.softmax)


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

feed={enLayer.activations:enData, deLayer.activations:deData}

print "Untrained outputs:"
print sess.run(readout.activations, feed_dict=feed)

cost = tf.nn.softmax_cross_entropy_with_logits(readout.weightedInputs, targets)
train = tf.train.GradientDescentOptimizer(1e-1).minimize(cost)

for i in range(50):
    sess.run(train, feed_dict=feed)

print "Trained outputs:"
print sess.run(readout.activations, feed_dict=feed)
