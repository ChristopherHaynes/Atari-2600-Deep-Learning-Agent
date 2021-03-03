import csv
import os.path
import datetime

# Manages file output, and saving results to csv files
class ResultsRecorder():
    def __init__(self, gameName):
        self.gameName = gameName
   
        # Record a timestamp for naming the results file
        self.dt = str(datetime.date.today())

        # Define the path of the results file
        self.resultsPath = './Results/' + gameName + '-' + self.dt + '.csv'
        fileExists = os.path.isfile(self.resultsPath)
        count = 1;
        while fileExists:         
            self.resultsPath = './Results/' + gameName + '-' + self.dt + '-' + str(count) + '.csv'
            fileExists = os.path.isfile(self.resultsPath)
            count = count + 1

        # Create a file with the timedate stamp as the name
        with open(self.resultsPath, 'w+', newline='') as csvResults:
            writer = csv.writer(csvResults, delimiter=',', dialect='excel')
            writer.writerow([str(self.gameName)])
            writer.writerow(['Episode Number', 'Episode Steps', 'Reward', 'Score', 'Epsilon'])

    def addEpisode(self, epNumber, episodeSteps, reward, score, epsilon):
        # Add the results of an episode to the results file
        with open(self.resultsPath, 'a', newline='') as csvResults:
            writer = csv.writer(csvResults, delimiter=',', dialect='excel')
            writer.writerow([epNumber, episodeSteps, reward, score, epsilon])