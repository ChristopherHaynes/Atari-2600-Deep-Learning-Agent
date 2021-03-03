import argparse

class ArgParser():

    # List of all compatable game names
    gameNames = ['Alien', 'Asterix', 'Asteroids',  'Atlantis', 'BattleZone', 'BeamRider', 
                 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 
                 'Defender', 'DemonAttack', 'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway',
                 'Frostbite', 'Gravitar', 'Hero',  'IceHockey', 'Jamesbond', 'Kaboom', 'Kangaroo',
                 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix',
                 'Pitfall', 'Pong', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
                 'Seaquest', 'Solaris', 'SpaceInvaders', 'StarGunner', 'TimePilot', 'UpNDown', 
                 'Venture','YarsRevenge', 'Zaxxon']

    def __init__(self):
        # Create an argument parser for performing an experiment on the agent
        self.parser = argparse.ArgumentParser(description='Determining Learning Parameters')
       
        # Add a series of arguments and flags to the parser
        self.parser.add_argument('-gn', '--gameName', 
                            help='The name of the game to train')
        self.parser.add_argument('-ec', '--episodeCount', 
                                help='The number of the final episode to be ran in this trial', 
                                type=int,
                                default=10000)
        self.parser.add_argument('-r', '--render',
                                help='Should the game be rendered to the screen?', 
                                default=False, 
                                action='store_const', 
                                const=True)
        self.parser.add_argument('-nt', '--noTrain', 
                                help='Should the model be trained?', 
                                default=True, 
                                action='store_const',
                                const=False)
        self.parser.add_argument('-s', '--save', 
                                 help='Should the trained model be saved?', 
                                 default=False, 
                                 action='store_const',
                                 const=True)
        self.parser.add_argument('-l', '--load', 
                                 help='Load a trained model by episode number (found on the model checkpoint filename)', 
                                 type=int, 
                                 default=0)

        # Collect the argunments returned by the parser
        self.args = self.parser.parse_args()

        # Ensure a valid game is loaded
        validName = False
        for i in range(0, len(self.gameNames)):
            if self.args.gameName == self.gameNames[i]:
                validName = True

        # If the name is not valid, ask the user to input a new name
        while not validName:
            if self.args.gameName == None:
                print("\n\nTo start training please enter a game from the list below.")
            else:
                print('\n\nThe Game Name entered is not compatabile with this program, please enter one of the following game names.')
            print ('\nValid Game Names:\n', self.gameNames)
            self.args.gameName = input('\nGame Name:')
            for i in range(0, len(self.gameNames)):
                if self.args.gameName == self.gameNames[i]:
                    validName = True

    def getArgs(self):
        return self.args