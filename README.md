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
    conda install git
    conda update --all
    conda clean -a
    pip install git+https://github.com/Kojoley/atari-py.git
    pip install gym[atari]
    pip install --upgrade tensorflow
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
    python agent.py -gn Breakout -ec 1000 -r -s
    
## List of Compatible Atari 2600 Games
A full list of game names compatiblle with this program are; Alien, Asterix, Asteroids, Atlantis, BattleZone, BeamRider, Berzerk, Bowling, Boxing,
Breakout, ChopperCommand, CrazyClimber, Defender, DemonAttack, ElevatorAction, Enduro, FishinDerby, Freeway, FrostBite, Gravitar, Hero, IceHockey, 
JamesBond, Kangaroo, Kaboom, Krull, KungFuMaster, MontezumaRevenge, MsPacman, NameThisGame, Phoenix, Pitfall, Pong, PrivateEye, Qbert, Riverraid,
RoadRunner, Robotank, Seaquest, Solaris, SpaceInvaders, StarGunner, TimePilot, UpNDown, Venture, YarsRevenge, Zaxxon.

## Default Hyper Parameters

## Neural Network Design

## Accuracy and Test Results

## Additional Details



