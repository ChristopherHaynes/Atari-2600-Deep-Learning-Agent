import tensorflow as tf

class NeuralNet:
    def __init__(self):
        inputs =  tf.placeholder(shape=[1, 16], dtype=tf.float32)
        weights =  tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
        netRecognition = tf.matmul(inputs, weights)
        softmaxOutput =  tf.argmax(netRecognition, 1)

        nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ, netRecognition))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        updateModel = trainer.minimize(loss)
        pass