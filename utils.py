import tensorflow as tf

def getSampledIds(labels, numSamples, numTargets, sampleFnc=tf.nn.learned_unigram_candidate_sampler):
    labels = tf.cast(labels, tf.int64)
    sampled = sampleFnc(labels, 1, numSamples, unique=True, range_max=numTargets)
    return labels, tf.stop_gradient(sampled[0])

def computeLogitsFromSampledEmbeddings(trueEmbeddings, sampledEmbeddings, labels, sampled, inputs, numSamples):

    # inputs: [batch, inDim]    sampledEmbeddings: [numSampled, inDim]
    #   sampledLogits: [batch, numSampled]
    sampledLogits = tf.matmul(inputs, sampledEmbeddings, transpose_b=True)
    # true logits are dot product of each input in the batch with its corresponding target embedding
    #   trueLogits: [batch, 1]
    trueLogits = tf.reduce_sum(tf.multiply(inputs, trueEmbeddings), axis=1, keep_dims=True)

    # Remove accidental target hits in the negative samples
    accHits = tf.nn.compute_accidental_hits(labels, sampled, num_true=1)
    batchRow, sampleCol, offsetWeights = accHits
    sparseIndices = tf.stack([batchRow, tf.to_int32(sampleCol)], 1)
    sampledLogitsShape = tf.stack([tf.shape(labels)[0], numSamples], 0)
    sampledLogits += tf.sparse_to_dense(sparseIndices, sampledLogitsShape, offsetWeights,
        default_value=0.0, validate_indices=False)

    outLogits = tf.concat([trueLogits, sampledLogits], 1)
    return outLogits
