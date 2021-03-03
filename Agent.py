import gym
import tensorflow as tf
import numpy as np
import random

from PreProcessor import PreProcessor
from DeepQNetwork import DeepQnetwork
from ExperienceReplay import ExperienceReplay
from ResultsRecorder import ResultsRecorder
from ArgumentParser import ArgParser

# Parse command line arguments
argParser = ArgParser()
args = argParser.getArgs()

# Constant Program Parameters
RENDER_SCREEN = args.render          # Should the game be rendered to the screen (default = False)
FRAMES_PER_SECOND = 30               # Framerate for watching a trained model
TRAIN_MODEL = args.noTrain           # Should the model be trained (default = True)
SAVE = args.save                     # Should the model be saved (default = False)
LOAD_REF = args.load                 # Episode number of checkpoint to be loaded (0 or less will load nothing)
EPISODE_COUNT = args.episodeCount    # Number of episodes in a trial
GAME_NAME = args.gameName            # The name of the game being tested
PATH = "./DQNModels/" + GAME_NAME    # The directory path for saving the model

# Hyper-Parameters
alpha = 0.0001          # Learning rate (value network)
gamma = 0.99            # Discount rate
epsilonMin = 0.05       # Minimum odds of selecting a random action
epsilonMax = 1.0        # Maximum odds of selecting a random action
epsilonDelta = 0.000002 # The difference the epsilon value should move from max to min each step
updateFrequency = 4     # How often (in steps) the network should update
replaySize = 30000      # The number of experiences which the agent remembers (12gb RAM required for 160k)
batchSize = 32          # Size of a batch of experiences for training the networks
targetNetUpdate = 10000 # Number of steps between updating the target network to the value network

#-----------------------------------------------------------------------
# Trial - Run through a number of episodes, save data and record results
#-----------------------------------------------------------------------
# Trial scope variables
trialSteps = 0;         # Total number of steps taken in the current trail
epsilon = epsilonMax;   # Current epsilon value

# Initialise the environment
env = gym.make(GAME_NAME + 'Deterministic-v4')

# Prepares each frame to produce a viable network input
preProcessor = PreProcessor()

# Stores experiences as a tuple of [s, a, r, s']
memory = ExperienceReplay(replaySize)

# Records results and stores them in a csv file
resultsRec = ResultsRecorder(GAME_NAME)

# Q Value approximation and target networks, set with as many output nodes as viable actions
valueNetwork = DeepQnetwork(len(env.unwrapped.get_action_meanings()), alpha)
targetNetwork = DeepQnetwork(len(env.unwrapped.get_action_meanings()), alpha)
init = tf.global_variables_initializer()

# Create tensorflow model saver
saver = tf.train.Saver()

#-----------------------------------------------------------------------
# Start a tensorflow session
#-----------------------------------------------------------------------
with tf.Session() as sess:
    # Initialise the session, set the weights of the value network randomly 
    sess.run(init)
    
    # Load a model if specified in the parameters
    if LOAD_REF > 0:
        saver.restore(sess, PATH + '/model-' + str(LOAD_REF) + '.ckpt')

        # Set the epsilon value to an estimated level for the number of passed steps
        epsilon = epsilon - (LOAD_REF * epsilonDelta * 1500)
        if epsilon < epsilonMin: epsilon = epsilonMin

        # Create a list of all trainable variables in the graph (0-9 value, 10-19 target)
        allVariables = tf.trainable_variables()
        valueVariables = allVariables[0:9]
        targetVariables = allVariables[10:19]
        replaceTarget = [tf.assign(t, v) for t, v in zip(targetVariables, valueVariables)]

        # Run the operation to replace the target network variables with the value network variables
        sess.run(replaceTarget)  
       
    #-----------------------------------------------------------------------
    # Run through a fixed number of episodes
    #-----------------------------------------------------------------------
    for episode in range(LOAD_REF, EPISODE_COUNT):
    
        # Episode scope variables
        observation = env.reset()
        rewardSum = 0
        scoreSum = 0
        done = False 
        stepsTaken = 0
        state = None
        nextState = None

        # Store the last 4 frames in a buffer for the creation of a single state
        frames = []

        #-----------------------------------------------------------------------
        # Episode - take one step through the environment and contemplate actions
        #-----------------------------------------------------------------------
        while not done:                               
            # Render the game to the screen (disable for improved performace)
            if RENDER_SCREEN:
                # If not training, control the framerate for easier viewing
                if not TRAIN_MODEL:
                    time.sleep(1 / FRAMES_PER_SECOND)
                env.render()
                       
            # Reduce the epsilon value is not yet at the minimum
            if epsilon > epsilonMin:
                epsilon -= epsilonDelta 
            else:
                epsilon = epsilonMin
                              
            # Take a random action if enough frames have not been recorded to produce a valid state
            if stepsTaken < 4:
                # Select a random action from the list of viable actions
                action = env.action_space.sample() 
            else:                     
                # Concantenate 4 consectutive frames into a state array (105x80x4 after preprocessing)
                state = np.concatenate(frames, 2)                
                del frames[0]               
                              
                # Select an action using greedy epsilon and the value approxomation network  
                if random.random() > epsilon :   
                    action = int(sess.run(valueNetwork.greedyOutput, \
                                        feed_dict={valueNetwork.stateInput:[state]})) 
                else:
                    # Select a random action from the list of viable actions
                    action = env.action_space.sample()             

            # Step through the environment with the selected action and recored the observation, reward and terminal state
            observation, score, done, info = env.step(action)
            # Reduce the reward to its sign to make all rewards equal and for better continuity between games
            reward = np.sign(score)     
                
            # Perform pre-processing on the presented observation and store at the back of the frame list
            processedFrame = preProcessor.preprocessFrame(observation)
            processedFrame = np.expand_dims(processedFrame, axis=2)
            frames.append(processedFrame)           
            
            if stepsTaken > 4:               
                # Concantenate the updated 4 frame history into the next state
                nextState = np.concatenate(frames, 2)

                # Create an experience and store it in the replay buffer
                experience = np.reshape(np.array([state, action, reward, nextState]), [1,4])              
                memory.add(experience)  
                  
                #-----------------------------------------------------------------------
                # Update neural networks
                #-----------------------------------------------------------------------                       
                if stepsTaken % updateFrequency == 0 and TRAIN_MODEL:                                    

                    # Gather a batch of experiences for training from the replay buffer
                    trainingBatch = memory.sample(batchSize \
                                                  if len(memory.experienceBuffer) >= batchSize \
                                                  else len(memory.experienceBuffer))                                

                    # Get the highest Q values for the next states using the target network              
                    nextStatesMaxAction = sess.run(targetNetwork.networkOutput, \
                                                  feed_dict={targetNetwork.stateInput:np.stack(trainingBatch[:, 3])})     
                
                    # Reduce the array to contain only the maximum Q value for each input from the batch
                    nextStatesMaxAction = np.amax(nextStatesMaxAction, axis=1)

                    # Each target value is the recieved reward plus the discounted maximum Q Value for the next state
                    targets = trainingBatch[:, 2] + (gamma * nextStatesMaxAction)

                    # Update the network with our target values.
                    sess.run(valueNetwork.updateModel, \
                                feed_dict={valueNetwork.stateInput:np.stack(trainingBatch[:, 0]), \
                                        valueNetwork.target:targets, \
                                        valueNetwork.actions:trainingBatch[:, 1]})                 

            # Periodically set the target network to equal the value network
            if (trialSteps + stepsTaken) % targetNetUpdate == 0:           

                # Create a list of all trainable variables in the graph (0-9 value, 10-19 target)
                allVariables = tf.trainable_variables()
                valueVariables = allVariables[0:9]
                targetVariables = allVariables[10:19]
                replaceTarget = [tf.assign(t, v) for t, v in zip(targetVariables, valueVariables)]

                # Run the operation to replace the target network variables with the value network variables
                sess.run(replaceTarget)  
            
            # Sum the total reward, score and steps taken for an episode
            rewardSum += reward
            scoreSum += score
            stepsTaken = stepsTaken + 1

        # Save the model of DQNs after every episode (if enabled)
        if SAVE:
            saver.save(sess, PATH + '/model-'+str(episode) + '.ckpt')

        # Add the episode steps to trial steps
        trialSteps = trialSteps + stepsTaken

        # Save the episode results to the csv file
        resultsRec.addEpisode(episode, stepsTaken, rewardSum, scoreSum, epsilon)

        # Print score, steps per episode, current trial steps
        print("\nEpisode", episode, "finished after", stepsTaken+1, "timesteps")
        print("Reward was ", rewardSum, ", Score was ", scoreSum)
        print("Trial step count:", trialSteps)
        print('Epsilon', epsilon)   
       
# Closing Message
print('')
print(GAME_NAME, 'agent has been trained for', EPISODE_COUNT - LOAD_REF, 'steps.')
if SAVE:
    print('The agent has been saved with model number', EPISODE_COUNT, 'in the "DQNModels" folder')
print('Training Complete!')
            
            
    
    