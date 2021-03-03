import numpy as np

# Used to preprocesses frames before they are used as an input in the Deep Q Network
class PreProcessor():
    # Basic constructor
    def __init__(self):
        self = self

    # Convert the RGB 128 color frame to greyscale and store in a compact way
    def toGreyScale(self, frame):
        return np.mean(frame, 2).astype(np.uint8)

    # Downsample the frame by half to reduce the size from 210x160 to 105x80
    def halfDownsample(self, frame):
        # Slices the list in both directions to contain only even indices
        return frame[::2, ::2]

     # Perform a full preprocess on a frame
    def preprocessFrame(self, frame):
        frameGrey = self.toGreyScale(frame)
        processedFrame = self.halfDownsample(frameGrey)
        return processedFrame