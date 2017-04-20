import TFlibs.networkcrafter as nc
import PythonTools.charIntTools as cit
import numpy as np
import time
tf = nc.tf


trainingSequence = 'The quick brown fox jumps over the lazy dog. The dog doesn\'t care much, so she decides that next time the fox jumps over her, she\'s going to be asleep.'

# Convert the string to list of chars 
print trainingSequence

# Map chars to ints
trainingNums, char2Num = cit.intListFromString(list(trainingSequence))
num2Char = cit.swapDictionary(char2Num)
NUM_CHARS = len(num2Char)

# Target array and source array, offset by 1
targets = np.array(trainingNums)
sources = np.array([-1] + trainingNums[:-1])

# Create lists of source/target batches
BATCH_SIZE = 128
if sources.shape[0] > BATCH_SIZE:
    targetBatches = np.split(targets, range(BATCH_SIZE, sources.shape[0], BATCH_SIZE))
    sourceBatches = np.split(sources, range(BATCH_SIZE, sources.shape[0], BATCH_SIZE))
else:
    targetBatches = [targets]
    sourceBatches = [sources]

sess = tf.Session()

# The RNN
network = nc.Network()
network.inputLayer(NUM_CHARS, applyOneHot=True)
network.basicGRU(100, nLayers=2)
network.fullConnectLayer(NUM_CHARS, tf.nn.softmax)
network.buildGraph()
# Create a placeholder for target values. 
network.defineTargets(NUM_CHARS, applyOneHot=True)

# Cost and gradient descent
cross_entropy = -tf.reduce_sum(network.targetVals*tf.log(network.outLayer.activations))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)


sess.run(tf.global_variables_initializer())
start = time.time()

# Training
for i in range(200):

    for sources, targets in zip(sourceBatches, targetBatches):
        feed = network.getFeedDict(sources, 0.5, targets=targets, sequenceLengths=sources.shape)
        train_step.run(session=sess, feed_dict=feed)

    network.resetRecurrentHiddens(sess) 

    if i%20 == 0:
        for sources, targets in zip(sourceBatches, targetBatches):
            predictions = \
                np.argmax(network.forward(sess, sources, 1, sequenceLengths=sources.shape), axis=1)
            print ''.join([num2Char[num] for num in predictions])

        network.resetRecurrentHiddens(sess) 
print time.time()-start
