import networkcrafter as nc
import numpy as np
tf = nc.tf

trainingSequence = ['t', 'h', 'e', ' ', 'q', 'u', 'i', 'c', 'k', ' ', 'b', 'r', 'o', 'w', 'n', ' ', 'f', 'o', 'x', ' ', 'j', 'u', 'm', 'p', 's', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'h', 'e', ' ', 'l', 'a', 'z', 'y', ' ', 'd', 'o', 'g']

NUM_CHARS = 28

trainingNums = []
[trainingNums.append(ord(letter)) for letter in trainingSequence]

targets = np.array(trainingNums[1:])
targets = targets-97
targets = np.where(targets<0, 26, targets)
targets = np.concatenate([targets, [27]])

sources = targets[:-1]
sources = np.concatenate([[19], sources])

onehot_targets = np.zeros([len(targets), NUM_CHARS])
for i in range(len(targets)):
    onehot_targets[i,targets[i]] = 1

onehot_sources = np.zeros([len(sources), NUM_CHARS])
for i in range(len(sources)):
    onehot_sources[i,sources[i]] = 1


# The RNN
inLayer = nc.InputLayer(NUM_CHARS)
rnnLayer = nc.RNN(inLayer, 300, NUM_CHARS, tf.nn.softmax)

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
                
        answer_fed = ['t']
        for j in range(targets.shape[0]):
            
            source = onehot_sources[j]
            source.shape = 1, source.shape[0]

            answer_fed.append(chr(97+np.argmax(sess.run(rnnLayer.activations, feed_dict={inLayer.activations:source}))))

        print ''.join(answer_fed).replace('{', ' ')
    
    #train_step.run(feed_dict={inLayer.activations:onehot_sources, y_:onehot_targets})
    rnnLayer.resetHiddenState()



answer_fed = ['t']
for i in range(targets.shape[0]):
    
    source = onehot_sources[i]
    source.shape = 1, source.shape[0]

    answer_fed.append(chr(97+np.argmax(sess.run(rnnLayer.activations, feed_dict={inLayer.activations:source}))))

print ''.join(answer_fed).replace('{', ' ')

rnnLayer.resetHiddenState()

predict_fed = ['t']
last_predict = onehot_sources[0]
last_predict.shape = 1, last_predict.shape[0]
for i in range(targets.shape[0]):

    predict = sess.run(rnnLayer.activations, feed_dict={inLayer.activations:last_predict})
    last_predict = predict

    predict_fed.append(chr(97+np.argmax(predict)))

print ''.join(predict_fed)


