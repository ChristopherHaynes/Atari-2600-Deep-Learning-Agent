import unittest
import tensorflow as tf
import numpy as np

from DeepQNetwork import DeepQnetwork
from ExperienceReplay import ExperienceReplay
from PreProcessor import PreProcessor
from ResultsRecorder import ResultsRecorder

# Test the functionality of the Deep Q Network
class TestDQN(unittest.TestCase):

    # Ensure that only inputs of the correct dimsonality and type are accepted
    def test_invalidInput(self):
        testNetwork = DeepQnetwork(6, 0.0001)
        init = tf.global_variables_initializer()

        # Start a tensorflow session so the test network can be examined
        with tf.Session() as sess:
            # Init the network 
            sess.run(init)
            
            # Define a three inputs, one valid, two invalid
            validInput = np.zeros((105, 80, 4), dtype=float)
            wrongSizeInput = np.zeros((210, 105, 3), dtype=float)
            wrongTypeInput = np.zeros((105, 80, 4), dtype=str)

            # Make assertions on the network output for each of the inputs
            self.assertIsNotNone(sess.run(testNetwork.greedyOutput, 
                                          feed_dict={testNetwork.stateInput:[validInput]}))

            # Should raise ValueError, placeholder is 105 x 80 x 4
            with self.assertRaises(ValueError):
                sess.run(testNetwork.greedyOutput, 
                         feed_dict={testNetwork.stateInput:[wrongSizeInput]})

            # Should raise ValueError, string can not explicitly be converted to float
            with self.assertRaises(ValueError):
                sess.run(testNetwork.greedyOutput, 
                         feed_dict={testNetwork.stateInput:[wrongTypeInput]})


    # Ensure the network has correct dimensonality at all points
    def test_networkShape(self):
        testNetwork = DeepQnetwork(6, 0.0001)
        init = tf.global_variables_initializer()

        # Start a tensorflow session so the test network can be examined
        with tf.Session() as sess:
            # Init the network 
            sess.run(init)
            
            # Define a valid input
            validInput = np.zeros((105, 80, 4), dtype=float)

            # Feed the input and gather the output at each layer
            networkOutputs = sess.run([testNetwork.conv1, testNetwork.conv2, testNetwork.conv3, testNetwork.dense], 
                                       feed_dict={testNetwork.stateInput:[validInput]})
            
            # Check the shape of every layer is as intended
            self.assertEqual(networkOutputs[0].shape, (1, 25, 19, 32))
            self.assertEqual(networkOutputs[1].shape, (1, 11, 8, 64))
            self.assertEqual(networkOutputs[2].shape, (1, 9, 6, 64))
            self.assertEqual(networkOutputs[3].shape, (1, 512))

    # Ensure that batches of n-size can be passed to the network
    def test_inputBatches(self):
        testNetwork = DeepQnetwork(6, 0.0001)
        init = tf.global_variables_initializer()

        # Start a tensorflow session so the test network can be examined
        with tf.Session() as sess:
            # Init the network 
            sess.run(init)

            # Define some valid batch inputs
            smallBatch = np.zeros((3, 105, 80, 4), dtype=float)
            bigBatch = np.zeros((32, 105, 80, 4), dtype=float)

            # Check the first conv layer to ensure the batch is being carried through
            smallOutput = sess.run(testNetwork.conv1, 
                                     feed_dict={testNetwork.stateInput:smallBatch})
            bigOutput = sess.run(testNetwork.conv1, 
                                     feed_dict={testNetwork.stateInput:bigBatch})

            # First dimension should be equal to the number of states in a batch
            self.assertEqual(smallOutput.shape, (3, 25, 19, 32))
            self.assertEqual(bigOutput.shape, (32, 25, 19, 32))

    # Ensure the network behaviour is consistent
    def test_networkRecognition(self):
        testNetwork = DeepQnetwork(50, 0.0001)
        init = tf.global_variables_initializer()

        # Start a tensorflow session so the test network can be examined
        with tf.Session() as sess:
            # Init the network 
            sess.run(init)
            
            # Define a two contrasting inputs
            zerosInput = np.zeros((105, 80, 4), dtype=float)
            onesInput = np.ones((105, 80, 4), dtype=float)

            # Retrieve an estimate from the network for both inputs
            zerosOutput = sess.run(testNetwork.greedyOutput, 
                                   feed_dict={testNetwork.stateInput:[zerosInput]})
            onesOutput = sess.run(testNetwork.greedyOutput, 
                                   feed_dict={testNetwork.stateInput:[onesInput]})

            # Assert that these two outputs should not be the same
            self.assertNotEqual(zerosOutput, onesOutput)

# Test the functionality of the Experience Repaly
class TestExperienceReplay(unittest.TestCase):

    # Test whether the replay can initalised to a range of sizes
    def test_variableSize(self):

        # Declare a range of experience replay buffers
        tinyER = ExperienceReplay(1)
        midER = ExperienceReplay(30000)
        bigER = ExperienceReplay(1000000)

        # Check the size of the deques
        self.assertNotEqual(len(tinyER.experienceBuffer), 1)
        self.assertNotEqual(len(midER.experienceBuffer), 30000)
        self.assertNotEqual(len(bigER.experienceBuffer), 1000000)

    # Ensure that the deque will not grow above the defined maximum
    def test_maxSize(self):

        # Declare a couple of experience replay buffers
        oneER = ExperienceReplay(1)
        fiveER = ExperienceReplay(5)

        # Overfill both buffers 
        dummyData = np.zeros((1,4), dtype=float)
        oneER.add(dummyData)
        oneER.add(dummyData)

        for i in range(0, 8):
            fiveER.add(dummyData)

        # Compare the actual buffer size to expected max
        self.assertEqual(len(oneER.experienceBuffer), 1)
        self.assertEqual(len(fiveER.experienceBuffer), 5)

    # Ensure that the oldest experiences are overwritten when a new one is added
    def test_firstInFirstOut(self):

        # Declare an experience replay buffer
        testER = ExperienceReplay(4)

        # Add incrementing numbers to fill the array
        for i in range(0, 4):
            testER.add(i)

        # Add a 10 to the array and check that the deque shuffles as expected
        testER.add(10)

        # First position in the deque should be lost, last position should be the new addition
        self.assertEqual(testER.experienceBuffer[0], 1)
        self.assertEqual(testER.experienceBuffer[1], 2)
        self.assertEqual(testER.experienceBuffer[2], 3)
        self.assertEqual(testER.experienceBuffer[3], 10)

        # Add a 20 to the array and ensure the shuffling continues
        testER.add(20)

        # First position in the deque should be lost, last position should be the new addition
        self.assertEqual(testER.experienceBuffer[0], 2)
        self.assertEqual(testER.experienceBuffer[1], 3)
        self.assertEqual(testER.experienceBuffer[2], 10)
        self.assertEqual(testER.experienceBuffer[3], 20)

    # Test that a correct sized batch is returned when requested
    def test_sampleBatch(self):

        # Declare an experience replay buffer
        testER = ExperienceReplay(4)

        # Create a sample test experience
        s = np.ones((105, 80, 4), float)
        a = 1
        r = 1
        ns = np.zeros((105, 80, 4), float)
        testExperience = np.reshape(np.array([s, a, r, ns]), [1,4]) 

        # Fill the buffer with test inputs
        for i in range(0, 4):
            testER.add(testExperience)

        # Request batches of varying sizes
        oneBatch = testER.sample(1)
        twoBatch = testER.sample(2)
        fourBatch = testER.sample(4)

        # Assert the shape of the returned batch
        self.assertEqual(oneBatch.shape, (1, 4))
        self.assertEqual(twoBatch.shape, (2, 4))
        self.assertEqual(fourBatch.shape, (4, 4))

# Test that frames are being correctly modified in the preprocessing stage
class TestPreProcessor(unittest.TestCase):

    # Greyscale tests
    def test_greyscale(self):

        # Create a preprocessor object and define a test RGB array (8-bit)
        preprocess = PreProcessor()
        testObservation = np.random.randint(255, size=(210, 160, 3))

        # Use the PP to convert to greyscale
        greyTest = preprocess.toGreyScale(testObservation)

        # Ensure the dimsonality has reduced
        self.assertEqual(greyTest.shape, (210, 160))

        # Ensure the greyscale has been applied correctly
        expectedGreyValue = int((testObservation[0, 0, 0] + testObservation[0, 0, 1] + testObservation[0, 0, 2]) / 3)
        self.assertEqual(expectedGreyValue, greyTest[0, 0])

    # Downsample tests
    def test_downsample(self):

        # Create a preprocessor object and define a test greyscale array (8-bit)
        preprocess = PreProcessor()
        testGreyFrame = np.random.randint(255, size=(210, 160))

        # Use the PP to half the size of the input
        halfFrame = preprocess.halfDownsample(testGreyFrame)

        # Ensure the frame has reduced in size by half
        self.assertEqual(testGreyFrame.shape[0]/2, halfFrame.shape[0])
        self.assertEqual(testGreyFrame.shape[1]/2, halfFrame.shape[1])

# Run the DQN Tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestDQN)
unittest.TextTestRunner(verbosity=2).run(suite)

# Run the Experience Replay Tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestExperienceReplay)
unittest.TextTestRunner(verbosity=2).run(suite)

# Run the Preprocessor Tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestPreProcessor)
unittest.TextTestRunner(verbosity=2).run(suite)

print('\nTesting Complete\n')