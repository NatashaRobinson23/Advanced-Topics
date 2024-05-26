#!/usr/bin/env python

# Library Imports
from inspect import Parameter
from unittest.mock import DEFAULT
import torch
import time
import random
import matplotlib.pyplot as plt

# Main Class for Particle Swarm Optimization
class ParticleSwarmOptimizer(object):
    def __init__(self,swarmSize=10,c_cognitive=2, c_social=2, inertiaWeight=0.1, velocityLimit=0.1, maxIterations=100, roles=['Police', 'Civilian', 'Fire'], timeSlots=torch.linspace(0, 24, steps=10), geographical=['Region1','Mountain','Region2'], upperBound=2000, lowerBound=0):
        self.swarmSize = swarmSize
        self.c_cognitive = c_cognitive # Amount the particle is influenced by the personal best. 
        self.c_social = c_social # Amount the particle is influenced by the global best.
        self.inertiaWeight = inertiaWeight # Influence of the current velocity on its future velocity. 
        self.velocityLimit = velocityLimit # How fast the particles move within the space. 
        self.maxIterations = maxIterations # How many times the code runs. 
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        
        # Addressing the requirements. 
        self.roles = roles
        self.timeSlots = timeSlots
        self.geographical = geographical
        self.userList = [[random.randint(0, 2) for temp in range(self.swarmSize)]]
        self.userTensor = torch.tensor(self.userList)
        
    def visualise(self):
        # Showing Particle Graphs. 
        for graph in self.graphs:
            graph.showPlot()
            
    def initialiseGraphs(self):
        # Initialising Scatter Plot Graphs
        self.graphs = []        
        for role in self.roles:
            self.graphs.append(ScatterPlot(role, self.maxIterations))  
        
    # Objective Function. Creates a scoring system based on the constraints of the scenario. 
    def objectiveFunction(self, frequency):
        
        # Detering away from 550-650. Forbidden frequencies - distress beacons. 
        restrictedPenalty = torch.zeros((1, self.swarmSize))
        restrictedMask = ((frequency >= 550) & (frequency <= 650))
        restrictedPenalty[restrictedMask] = 10**10
        print("Restricted Penalty: ", restrictedPenalty)
        
        # Proximity Evaluation. 
        # Punish based on how far the particle is from the average of other particles of the same group.
        proximityPenalty = torch.zeros((1, self.swarmSize))
        repulsionReward = torch.zeros((1, self.swarmSize))
        uniqueTypes = torch.unique(self.userTensor) # The different types possible (0 Police, 1 Civilian, 2 Fire). 

        for index, userType in enumerate(uniqueTypes):
            frequenciesType = frequency[self.userTensor == userType] # Selects frequencies which are of the userType.            
            typeMean = torch.mean(frequenciesType)
            absDiff = torch.abs(frequenciesType - typeMean)
            proximityPenalty[self.userTensor == userType] = torch.square(absDiff) 
                                 
            # Print Statements
            print("User Tensor: ", self.userTensor)
            print("Frequencies of the Current Type: ", userType, " ", frequenciesType)
            print("Centre of Distribution: ", typeMean)
            print("Proximity Penalty: ", proximityPenalty)
            
            # Appending to the graphs based on types. 
            for i in range(0, len(frequenciesType)):
                self.graphs[index].append(frequenciesType.numpy()[i], i)    
            
            # Repulsion Evaluation. 
            # Repels certain groups from placing near one another.           
            for otherUserType in uniqueTypes:
                if otherUserType != userType:
                    otherFrequencyType = frequency[self.userTensor == otherUserType] # Make a tensor for each group type. 
                    otherTypeMean = torch.mean(otherFrequencyType) # Calculate the mean of the group type. 
                    otherAbsDiff = torch.abs(frequenciesType - otherTypeMean) # Calculate the absolute difference between frequencies and other group mean.
                    repulsionReward[self.userTensor == userType] += torch.square(otherAbsDiff) # Append repulsion score. PSO will seek to maximise this value. 

                    print(repulsionReward)

            print("Repulsion Penalty: ", repulsionReward)                
        totalPenalty = restrictedPenalty + (proximityPenalty) - (repulsionReward) # It is trying to minimise this. 
        print("Total Penalty: ", totalPenalty)
        return totalPenalty

    def search_space(self):
        self.upperBound = torch.tensor([self.upperBound])
        self.lowerBound = torch.tensor([self.lowerBound])
        self.dimensionality = self.upperBound.size()[0]    

    def populate(self):
        self.frequency = ((self.upperBound - self.lowerBound)*torch.rand(1,self.swarmSize)) + self.lowerBound
        print("starting: ", self.frequency)
        self.velocity = (2*self.velocityLimit*torch.rand(self.dimensionality,self.swarmSize)) - self.velocityLimit
        print("starting2: ", self.velocity)

    def enforce_bounds(self):
        upperBound = self.upperBound.view(self.dimensionality,1)
        lowerBound = self.lowerBound.view(self.dimensionality,1)
        self.frequency = torch.max(torch.min(self.frequency,upperBound),lowerBound)
        self.velocity = torch.max(torch.min(self.velocity,torch.tensor(self.velocityLimit)),-torch.tensor(self.velocityLimit))

    def run(self,verbosity = True):
        self.currentFitness = self.objectiveFunction(self.frequency) # Evaluates the fitness function at each particle position. 
        self.particleBest = self.frequency
        
        self.particleBestFitness = self.currentFitness
        self.globalBest = self.frequency[:,self.particleBestFitness.argmin()].view(self.dimensionality,1)  # Picks particle with the lowest penalty. 
        print("Inital Global Best: ", self.globalBest)
        self.globalBestFitness = self.particleBestFitness.min()
        for iteration in range(self.maxIterations):
            tic = time.monotonic()
            
            self.velocity = ((self.inertiaWeight*self.velocity) +  # Current velocity times some weight
                            (self.c_cognitive*torch.rand(1)*(self.particleBest-self.frequency)) + # plus how far it is from its personal best times a random number. 
                            (self.c_social*torch.rand(1)*(self.globalBest-self.frequency)))
            print("Velocity: ", self.velocity)
            self.frequency = self.frequency + self.velocity
            self.enforce_bounds()
            self.currentFitness = self.objectiveFunction(self.frequency) # Penalty function. 
            localMask = self.currentFitness<self.particleBestFitness # Boolean, true when current fitness less than particle best fitness. 
            self.particleBest[localMask] = self.frequency[localMask] # Updates the personal best positions for those only who were true. 
            self.particleBestFitness[localMask] = self.currentFitness[localMask] # Updates particle best fitness values for those who were true. 
            
            # If the fitness function is less than the global best fitness, update the global best fitness. 
            if (self.currentFitness.min() < self.globalBestFitness):
                self.globalBestFitness = self.currentFitness.min()
                self.globalBest = self.frequency[:,self.currentFitness.argmin()].view(self.dimensionality,1)
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f} | current mean fitness {:.3f} | iteration time {:.3f}'
                .format(iteration + 1,self.globalBestFitness,self.currentFitness.mean(),toc-tic))
                print("Final Frequency: ", self.frequency)
                
class ScatterPlot:
    def __init__(self, type, iterations):
        self.type = type
        self.data = {'x': [], 'y': list(range(0, iterations+1))}
        self.colors = ['red','orange','yellow','green','blue','purple','pink']
            
    def append(self, x, index):
        while index >= len(self.data['x']):
            self.data['x'].append([])
        self.data['x'][index].append(x)
            
    def showPlot(self):
        for i in range(0, len(self.data['x'])):
            plt.plot(self.data['x'][i], self.data['y'], color=self.colors[i], label=f"Particle {i+1}: {round(self.data['x'][i][-1])}")
        plt.ylabel('Iteration')
        plt.xlabel('Frequency')
        plt.title(self.type)
        plt.grid(True)
        plt.legend()
        plt.show()
                    
    def colorCheck(self, userTensor, index):
        return self.colors[userTensor[0][index]]
        
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    print("|===============================================================|")
    print("|Default Parameters:                                            |")
    print("|[0]  swarmSize: 10                                             |")
    print("|[1]  c_cognitive: 2                                            |")
    print("|[2]  c_social: 2                                               |")
    print("|[3]  inertiaWeight: 0.1                                        |")
    print("|[4]  velocityLimit: 0.1                                        |")
    print("|[5]  maxIterations: 100                                        |")
    print("|[6]  roles (randomly assigned): Police, Civilian and Fire.     |")
    print("|[7]  timeSlots: 0-24, <swarm size> steps.                      |")
    print("|[8]  geographical: Region1, Mountain, Region2.                 |")
    print("|[9]  upperBound: 2000                                          |")
    print("|[10] lowerBound: 0                                             |")
    print("|===============================================================|")
    
    p = ParticleSwarmOptimizer()
    inputBool = str(input("Would you like to edit any of the default parameters? [y/n]: "))
    if inputBool == 'y':
        while inputBool != 'n':
        
            choice = int(input("Which parameter would you like to change? (Pick num. 0-10): "))

            match choice:
                case 0:
                    p.swarmSize = int(input("Enter swarm size: "))
                case 1: 
                    p.c_cognitive = float(input("Enter c_cognitive: "))
                case 2:
                    p.c_social = float(input("Enter c_social: "))
                case 3:
                    p.inertiaWeight = float(input("Enter inertia weight: "))
                case 4:
                    p.velocityLimit = float(input("Enter velocity limit: "))
                case 5:
                    p.maxIterations = int(input("Enter max iterations: "))
                case 6:
                    numRoles = int(input("Enter number of desired roles: "))
                    roles = []
                    for i in range(0, numRoles):
                        roles.append(input(f"Enter role {i+1}: "))
                    p.roles = roles
                case 7:
                    times = []
                    for i in range(0, p.swarmSize):
                        times.append(input(f"Enter time {i+1}: "))
                    p.timeSlots = times
                case 8:
                    numGeo = int(input("Enter number of desired regions: "))
                    geo = []
                    for i in range(0, numGeo):
                        geo.append(input(f"Enter geographical location {i+1}: "))
                    p.geographical = geo
                case 9:
                    upperBound = int(input("Enter upper bound: "))
                    p.upperBound = upperBound
                case 10:
                    lowerBound = int(input("Enter lower bound: "))
                    p.lowerBound = lowerBound
                
            inputBool = str(input("Do you want to change other parameters? [y/n]: "))
    
    p.initialiseGraphs()
    p.search_space()
    p.populate()
    p.run()
    p.visualise()
    

    
        # USE CASES:
        # FIRE: HF/VHF/UHF
        # EMERGENCY SERVICES: HF/VHF/UHF
        # POLICE: HF/UHF    
        
        # HF is more reliable at night. 
        # Mountains can obstruct HF signals. 
        # 
        # Frequency Capabilties.
        #timeReward = torch.zeros((1, SwarmSize))     
        #timeReliability = ((timeTensor >= 18) | (timeTensor <= 6))
        #timeReward[timeReliability] = 1000
        
        #locationPenalty = torch.zeros((1, SwarmSize))         