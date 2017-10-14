import tensorflow as tf

def getLogSampledIds(labels, numSamples, numTargets):
    labels = tf.cast(labels, tf.int64)
    sampled = tf.nn.learned_unigram_candidate_sampler(labels, 1, numSamples, unique=True,
        range_max=numTargets)
    return labels, sampled[0]

def computeLogitsFromSampledEmbeddings(trueEmbeddings, sampledEmbeddings, trueLabels,
    sampledLabels, inputs):

    # inputs: [batch, inDim]    sampledEmbeddings: [numSampled, inDim]
    #   sampledLogits: [batch, numSampled]
    sampledLogits = tf.matmul(inputs, sampledEmbeddings, transpose_b=True)

    # true logits are dot product of each input in the batch with its corresponding target embedding
    #   trueLogits: [batch, 1]
    trueLogits = tf.reduce_sum(tf.multiply(inputs, trueEmbeddings), axis=1)
    outLogits = tf.concat([trueLogits, sampledLogits], 1)
    return outLogits
