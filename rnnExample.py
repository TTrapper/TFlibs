import networkcrafter as nc
import numpy as np
tf = nc.tf

trainingSequence = 'the quick brown fox jumps over the lazy dog'
trainingSequence = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'
trainingSequence='abcdefghijklmnopqrstuvwxyz'
trainingSequence = list(trainingSequence)

NUM_CHARS = len(set(trainingSequence))+1
ASCII_OFFSET = 97 

trainingNums = []
[trainingNums.append(ord(letter)) for letter in trainingSequence]

targets = np.array(trainingNums[1:])
targets = targets-ASCII_OFFSET
targets = np.where(targets<0, NUM_CHARS-2, targets)
targets = np.concatenate([targets, [NUM_CHARS-1]])

sources = targets[:-1]
sources = np.concatenate([[trainingNums[0]-ASCII_OFFSET], sources])
print sources
onehot_targets = np.zeros([len(targets), NUM_CHARS])
for i in range(len(targets)):
    onehot_targets[i,targets[i]] = 1

onehot_sources = np.zeros([len(sources), NUM_CHARS])
for i in range(len(sources)):
    onehot_sources[i,sources[i]] = 1


# The RNN
inLayer = nc.InputLayer(NUM_CHARS)
rnnLayer = nc.RNN(inLayer, 10, NUM_CHARS, tf.nn.softmax)

y_ = tf.placeholder(tf.float32, shape=[None, NUM_CHARS])

cross_entropy = -tf.reduce_sum(y_*tf.log(rnnLayer.activations))
mean_square = tf.reduce_mean((y_-rnnLayer.activations)**2)

train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-3).minimize(mean_square)
#train_step = tf.train.GradientDescentOptimizer(1).minimize(mean_square)

correct_prediction = tf.equal(tf.argmax(rnnLayer.activations, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


for i in range(10000):
        
    for j in range(targets.shape[0]):
        source = onehot_sources[j]
        target = onehot_targets[j]
        source.shape = 1, source.shape[0]
        target.shape = 1, target.shape[0]
        
        train_step.run(feed_dict={inLayer.activations:source, y_:target})    

    rnnLayer.resetHiddenState()
    if i%10 == 0:
                
        answer_fed = [trainingSequence[0]]
        for j in range(targets.shape[0]):
            
            source = onehot_sources[j]
            source.shape = 1, source.shape[0]

            answer_fed.append(chr(ASCII_OFFSET+np.argmax(sess.run(rnnLayer.activations, feed_dict={inLayer.activations:source}))))

        print ''.join(answer_fed).replace('{', ' ')
    
    #train_step.run(feed_dict={inLayer.activations:onehot_sources, y_:onehot_targets})
    rnnLayer.resetHiddenState()

answer_fed = [trainingSequence[0]]
for i in range(targets.shape[0]):
    
    source = onehot_sources[i]
    source.shape = 1, source.shape[0]

    answer_fed.append(chr(ASCII_OFFSET+np.argmax(sess.run(rnnLayer.activations, feed_dict={inLayer.activations:source}))))

print ''.join(answer_fed).replace('{', ' ')

rnnLayer.resetHiddenState()

predict_fed = [trainingSequence[0]]
last_predict = onehot_sources[0]
last_predict.shape = 1, last_predict.shape[0]
for i in range(targets.shape[0]):

    predict = sess.run(rnnLayer.activations, feed_dict={inLayer.activations:last_predict})
    last_predict = predict

    predict_fed.append(chr(ASCII_OFFSET+np.argmax(predict)))

print ''.join(predict_fed)
