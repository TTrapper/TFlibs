import networkcrafter as nc
import numpy as np
tf = nc.tf

trainingSequence = 'The quick brown fox jumps over the lazy dog.'
trainingSequence = 'The quick brown fox jumps over the lazy dog. Then she read a book about the war and thought about her place in world and wondered: "Where do strawberries come from anyway?"'
#trainingSequence = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
#trainingSequence='abcdefghijklmnopqrstuvwxyz'
#trainingSequence='aaaaab'

# Add special START and STOP chars to the sequence
trainingSequence = list('#'+trainingSequence)
print trainingSequence

NUM_CHARS = len(set(trainingSequence))
# Map chars to ints
trainingNums = []
uniques = list(set(trainingSequence))
for letter in trainingSequence:
    trainingNums.append(uniques.index(letter))

# Dictionary gets us the chars back from ints
num2Char = dict(zip(trainingNums, trainingSequence))

# Target array and source array, offset by 1
targets = np.array(trainingNums[1:])
sources = np.array(trainingNums[:-1])

# One-hot representations
onehot_targets = np.zeros([len(targets), NUM_CHARS])
for i in range(len(targets)):
    onehot_targets[i,targets[i]] = 1
onehot_sources = np.zeros([len(sources), NUM_CHARS])
for i in range(len(sources)):
    onehot_sources[i,sources[i]] = 1

# Create lists of source/target batches
BATCH_SIZE = 50
if sources.shape[0] > BATCH_SIZE:
    onehot_targets = np.split(onehot_targets, range(0, sources.shape[0], BATCH_SIZE))
    onehot_sources = np.split(onehot_sources, range(0, sources.shape[0], BATCH_SIZE))
else:
    onehot_targets = [onehot_targets]
    onehot_sources = [onehot_sources]


# The RNN
inLayer = nc.InputLayer(NUM_CHARS)
rnnLayer = nc.RNN(inLayer, 100, NUM_CHARS, tf.nn.softmax)

y_ = tf.placeholder(tf.float32, shape=[None, NUM_CHARS])

cross_entropy = -tf.reduce_sum(y_*tf.log(rnnLayer.activations))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(rnnLayer.activations, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
writer = tf.train.SummaryWriter("./tensorlog", sess.graph)
sess.run(tf.initialize_all_variables())

for i in range(10000):
    for sources, targets in zip(onehot_sources, onehot_targets):

        feed = {inLayer.activations:sources, y_:targets}
        train_step.run(feed_dict=feed)

    if i%20 == 0:
        for sources, targets in zip(onehot_sources, onehot_targets):
            feed = {inLayer.activations:sources, y_:targets}
            predictions = np.argmax(sess.run(rnnLayer.activations, feed_dict=feed), axis=1)
            print ''.join([num2Char[num] for num in predictions]) 
