import numpy as np
import tensorflow as tf

class DeepQnetwork():
    # Define the structure of the convolutional neural network
    def __init__(self, outputSize, learningRate):

        # Create a placeholder for the input from the current state (4 greyscale frames)
        self.stateInput =  tf.placeholder(shape=[None,105,80,4],dtype=tf.float32)

        # Define the shape of a preprocessed input state [vary for batch size, height, width, number of frames]
        self.shapedInput = tf.reshape(self.stateInput,shape=[-1,105,80,4])

        # Define the convolution layers all with linear rectifier activation (ReLU)
        # First conv layer input (105x80x4), output (25x19x32)
        self.conv1 = tf.layers.conv2d( \
                            inputs=self.shapedInput, \
                            filters=32, \
                            kernel_size=[8,8], \
                            strides=[4,4], \
                            padding='VALID', \
                            activation=tf.nn.relu)      

        # Second conv layer input (25x19x32), output (11x8x64)
        self.conv2 = tf.layers.conv2d( \
                            inputs=self.conv1, \
                            filters=64, \
                            kernel_size=[4,4], \
                            strides=[2,2], \
                            padding='VALID', \
                            activation=tf.nn.relu)

        # Third conv layer input (11x8x64), output (9x6x64)
        self.conv3 = tf.layers.conv2d( \
                            inputs=self.conv2, \
                            filters=64, \
                            kernel_size=[3,3], \
                            strides=[1,1], \
                            padding='VALID', \
                            activation=tf.nn.relu)

        # Define the fully connected layers for feature classification
        # Flatten the input (9x6x64) down into a single dimension (3456)
        self.conv3Flat = tf.layers.flatten(self.conv3)

        # First dense layer input (3456), output (512)
        self.dense = tf.layers.dense( \
                                inputs=self.conv3Flat, \
                                units=512, \
                                activation=tf.nn.relu)

        # Output layer input (512), output (number of viable actions for selected game)
        self.networkOutput = tf.layers.dense( \
                                    inputs=self.dense, \
                                    units=outputSize)
        
        # Perform the softmax algorithm to recieve a probablisitc range across all output nodes
        self.softmaxOutput = tf.nn.softmax(self.networkOutput)
        # Takes the action with the highest predicted Q value
        self.greedyOutput = tf.argmax(input=self.networkOutput, axis=1, output_type=tf.int32)
  
        # Define a loss function for the network - This function is the sum of the squared error
        # Where (error = target output - actual output)
        self.target = tf.placeholder(dtype=tf.float32)
        
        # Define a one-hot action mask for training
        self.actions = tf.placeholder(dtype=tf.int32)
        self.actionsOnehot = tf.one_hot(self.actions, outputSize, dtype=tf.float32)

        # Define the single Q value for the chosen action by applying the one-hot action mask
        # As the mask will set all other outputs to 0, the array can then be summed for the chosen action Q value.
        self.actual = tf.reduce_sum(tf.multiply(self.networkOutput, self.actionsOnehot), axis=1)
              
        # Calculate the square of the error, reduce mean the array 
        self.error = tf.square(self.target - self.actual)
        self.sumSquareError = tf.reduce_mean(self.error)

        # Determine the trainer for stocastic gradient decent
        self.trainer = tf.train.AdamOptimizer(learning_rate=learningRate)
        # Define the function to update the model using the trainer to reduce the sum squared error.
        self.updateModel = self.trainer.minimize(self.sumSquareError)
       
