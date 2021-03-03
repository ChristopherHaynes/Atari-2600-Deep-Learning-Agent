import numpy as np
import collections
import random

# Holds a buffer of previous experiences (tuple of [s, a, r, s'])
class ExperienceReplay():
    # Create a buffer and define the maximum size on construction
    def __init__(self, maxSize = 50000):
        # Deque provides O(1) opperations when full, unlike list O(n)
        self.experienceBuffer = collections.deque(maxlen=maxSize)
     
    # Add the latest experience at the end of the buffer
    def add(self, experience):       
        # Deque will automatically overwrite oldest experience if the buffer is full
        self.experienceBuffer.append(experience)
  
    # Randomly sample a batch of experiences from the buffer
    def sample(self, batchSize):
        # Create a placeholder list for the batch
        trainingBatch = []
        # Select a random index (in range of the current buffer)
        for i in range(0, batchSize):
            # Add the randomly selected experience to the batch
            trainingBatch.append(self.experienceBuffer[random.randint(0, len(self.experienceBuffer) - 1)])
        # Convert the list to an array and reshape the dimensions to [batchSize, 4] where [:,4] = [s, a, r, s']
        return np.reshape(np.array(trainingBatch), [batchSize,4])
          