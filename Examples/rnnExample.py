import TFlibs.networkcrafter as nc
import PythonTools.charIntTools as cit
import numpy as np
tf = nc.tf

trainingSequence = 'The quick brown fox jumps over the lazy dog.'
#trainingSequence = 'The quick brown fox jumps over the lazy dog. The dog doesn\'t care much, so she decides that next time the fox jumps over her, she\'s going to be asleep. Later on, the fox did jump over the dog and, seeing no surprise in the in her whatsoever, felt annoyed. The fox, quick as it was, pounce on the dog and nipped her ear before sprinting off into the forest. Foxes are sly, but dogs are keen. The dog, now having been awaken from her slumber, had been keened indeed. It was time for a game. A game that not only used all of the letters, but would also decide who among them was the better hunter, hider, trapper, coniver.'
trainingSequence = 'The quick brown fox jumps over the lazy dog. The dog doesn\'t care much, so she decides that next time the fox jumps over her, she\'s going to be asleep.'
#trainingSequence = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
#trainingSequence='abcdefghijklmnopqrstuvwxyz'
#trainingSequence='aaaaab'

# Add a START character to the front of the sequence. Targets will begin with next character.
print trainingSequence
trainingSequence = list(trainingSequence)

# Map chars to ints
trainingNums, char2Num = cit.intListFromString(trainingSequence)
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

# The RNN
network = nc.Network()
network.inputLayer(NUM_CHARS, applyOneHot=True)
network.rnnLayer(100)
network.gruLayer(100)
network.fullConnectLayer(NUM_CHARS, tf.nn.softmax)

# Create a placeholder for target values. 
network.defineTargets(NUM_CHARS, applyOneHot=True)

# Cost and gradient descent
cross_entropy = -tf.reduce_sum(network.targetVals*tf.log(network.outLayer.activations))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

sess = tf.InteractiveSession()
writer = tf.train.SummaryWriter("./tensorlog", sess.graph)
sess.run(tf.initialize_all_variables())

# Training
for i in range(10000):
    for sources, targets in zip(sourceBatches, targetBatches):
        feed = network.getFeedDict(sources, targets=targets)
        train_step.run(feed_dict=feed)

    if i%20 == 0:
        for sources, targets in zip(sourceBatches, targetBatches):
            feed = network.getFeedDict(sources)
            predictions = np.argmax(sess.run(network.outLayer.activations, feed_dict=feed), axis=1)
            print ''.join([num2Char[num] for num in predictions])

    network.resetRecurrentHiddens()
