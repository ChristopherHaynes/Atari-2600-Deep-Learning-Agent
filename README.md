# Atari-2600-Deep-Learning-Agent
This program was created as my final year project for my undergraduate Computer Science degree. It is used for performing deep reinforcement learning
experiments on the Atari 2600. It can launch a range of game environments using AIGym, and then train agents how to play based on a pre-determined
set of hyper parameters. The user can determine several program parameters based on the experiment they wish to carry out.

## Program Requirements
The following software is required:
* Python 3.6 or later (Recommended: Anaconda for easy installation)
* *(Python Package)* Gym
* *(Python Package)* Tensorflow
* Microsoft Build Tools for Visual Studio 2015/2017

## Windows Installation Procedure
To prepare a Windows system to run this program, the following steps should be followed;
With Anaconda installed on the machine, open a CLI and enter the following commands;
```
conda install git
conda update --all
conda clean -a
pip install git+https://github.com/Kojoley/atari-py.git
pip install gym[atari]
pip install --upgrade tensorflow
```
If you have a NVidia GPU and want to optimise the performance of the program, then the GPU version of tensorflow should be installed. This requires
a few extra downloads and configuration steps which can be found in Tensorflow's documentation;
https://www.tensorflow.org/install

## Launching a Training Session
Download all the required project files and save them in a driectory of your choice. From a CLI navigate to the directory.
A session with default program parameters can be launched with:
    python agent.py
This will launch the program and ask the user which game they wish to test on. Type the name of the game you wish to test and press enter to begin.

## Command Line Arguments
For complex control of the program several command line arguments can be used to set certain parameters;

Name | Abbreviation | Description | Default Value
-----|--------------|-------------|--------------
--gameName | -gn | Name of the game to be loaded into the program | N/A
--episodeCount | -ec | The number of episodes to be ran in this trial | 10
--render | -r | Should the game be rendered to the screen | False
--noTrain | -nt | Should the model be trained | True
--save | -s | Should the model be saved | False
--load | -l | Load a trained model by episode number | 0

An example of using these arguments to launch an experiment on the game "Breakout" for 1000 episodes, with the game being rendered to the screen and
the model being saved would look like;
```
python agent.py -gn Breakout -ec 1000 -r -s
```
    
## List of Compatible Atari 2600 Games
A full list of game names compatiblle with this program are; Alien, Asterix, Asteroids, Atlantis, BattleZone, BeamRider, Berzerk, Bowling, Boxing,
Breakout, ChopperCommand, CrazyClimber, Defender, DemonAttack, ElevatorAction, Enduro, FishinDerby, Freeway, FrostBite, Gravitar, Hero, IceHockey, 
JamesBond, Kangaroo, Kaboom, Krull, KungFuMaster, MontezumaRevenge, MsPacman, NameThisGame, Phoenix, Pitfall, Pong, PrivateEye, Qbert, Riverraid,
RoadRunner, Robotank, Seaquest, Solaris, SpaceInvaders, StarGunner, TimePilot, UpNDown, Venture, YarsRevenge, Zaxxon.

## Default Hyper Parameters
Name | Value | Description 
-----|-------|------------
Alpha | 0.0001 | Learning rate for Adam algorithm
Gamma | 0.99 | Discount rate of subsequent states value
Epsilon Minimum | 0.05 | Lowest Value epsilon can reach
Epsilon Maximum | 1.0 | Highest, and start value of epsilon
Epsilon Delta | 0.000002 | Amount by which epsilon anneals each step
Replay Size | 150000 | Number of experiences stored in memory
Update Frequencey | 4 | Number of steps between network updates
Batch Size | 32 | Number of experiences sampled for one update
Target Update Frequency | 10000 | Number of steps between target network updates

## Deep Q Reinforcement Learning Design
This agent uses a range of different techniques to improve performance; seperate target and value networks, experience replay buffer and batch
updating being the main concepts. 

The neural network itself is a scaled down version of a similar structure suggest by deepmind. It is a convolutional neural network (CNN) with
the following structure;
![CNN structure](https://github.com/ChristopherHaynes/Atari-2600-Deep-Learning-Agent/blob/master/res/nn_structure.png?raw=true)

To improve performance there is also pre-processing performed on each frame, reducing it from RGB to grey-scale, down-sampling and then 
stacking 4 consective frame into a 3D tensor;
![Preprocessing](https://github.com/ChristopherHaynes/Atari-2600-Deep-Learning-Agent/blob/master/res/preprocessing.png?raw=true)

Further detail on the design, rationale and mathematics can be found in the full project report which is included in the __res__ directory.

## Accuracy and Test Results
Overall this agent can achieve super-human results in some games (Pong and Breakout) and achieve some level of learning in most others.
Full tests on all the available games have not yet been carried out, but the results of 5,000,000 steps in 5 different games can be seen
here;
![Learning Results Graphs](https://github.com/ChristopherHaynes/Atari-2600-Deep-Learning-Agent/blob/master/res/results_graphs.png?raw=true)

## Additional Details
There is also additonal options to run a variation of the standard agent to test two different novel methods for varying the epsilon
value whilst training; Stepped Annealing Epsilon (SAE) and Variable Epsilon (VE). The instructions on how to test these methods and a write up
of their performance can be found in the full project report which is included in the __res__ directory.


