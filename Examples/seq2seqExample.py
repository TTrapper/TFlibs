import numpy as np
import networkcrafter as nc
import tensorflow as tf
import PythonTools.charIntTools as cit

someText = "This is a random bit of words that we will attempt train on. For this example, we are not concerned about overfitting the data. In fact, the network will have so little information that it will likely need to simply memorize the answers as best it can. The goal is to get a proof-of-concept for some curiosity's sake."

nums, chr2Int = cit.intListFromString(someText)
int2Char = cit.swapDictionary(chr2Int)

someText = []
word = []
for n in nums:
    if int2Char[n] is not ' ':
        word.append(n)
    else:
        someText.append(word)
        word = []
someText.append(word)

print someText

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