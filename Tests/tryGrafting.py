import tensorflow as tf
import numpy as np
import networkcrafter as nc


sess = tf.InteractiveSession()

inputs = tf.ones([2,10])

inLayer = nc.InputLayer(10)
hLayer = nc.FullConnectLayer(inLayer, 100, tf.nn.relu)
outLayer = nc.FullConnectLayer(hLayer, 2, tf.nn.softmax)

inLayer.activations = inputs

hLayer.buildGraph()
outLayer.buildGraph()


sess.run(tf.initialize_all_variables())

print sess.run(outLayer.activations)


net1 = nc.Network()
net1.inputLayer(10)
net1.fullConnectLayer(20, tf.nn.relu)

net2 = nc.Network()
net2.inputLayer(20)
net2.fullConnectLayer(20, tf.nn.relu)
net1.graftOn(net2)

sess.run(tf.initialize_all_variables())
net1.buildGraph()

#The following three printouts should be identical
print net1.forward(sess, np.ones([2,10]))
net1.setInputs(inputs)
net1.buildGraph()
print net1.forward(sess, 2*np.ones([2,10]))
print sess.run(net1.outputs)


nNodes = 100
nOut = 2
nIn = 10
enLen = 2
deLen = 3

readout = nc.Network()
readout.inputLayer(nNodes)
readout.fullConnectLayer(nOut, tf.nn.softmax)

seq2seq = nc.Network()
seq2seq.inputLayer(nIn)
seq2seq.defineDecodeInLayer(nOut)
seq2seq.seq2SeqBasic(nNodes,enLen,deLen,wb=readout)
readout.buildGraph()
seq2seq.graftOn(readout)


seq2seq.buildGraph()

sess.run(tf.initialize_all_variables())

print seq2seq.forward(sess, np.ones([enLen, nIn]), decoderInputs=np.ones([deLen, nOut]))





