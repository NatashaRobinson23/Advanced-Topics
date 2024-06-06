#!/usr/bin/env python

# Library Imports
from inspect import Parameter
from unittest.mock import DEFAULT
import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda import is_available

# Checking if GPU is available and creating it as a device accordingly. 
if torch.cuda.is_available():
    GPUdevice = torch.device("cuda")
    ("GPU Available")

# Main Class for Particle Swarm Optimization
class ParticleSwarmOptimizer(object):
    def __init__(self):
        self.swarmSize = 10
        self.c_cognitive = 2 # Amount the particle is influenced by the personal best. 
        self.c_social = 2 # Amount the particle is influenced by the global best.
        self.inertiaWeight = 0.5 # Influence of the current velocity on its future velocity. 
        self.velocityLimit = 0.5 # How fast the particles move within the space. 
        self.maxIterations = 100 # How many times the code runs. 
        self.frequencyBounds = [np.log10(30000), np.log10(3000)] # Upper and lower frequency bounds. Default is the HF band.  
        
        # Addressing the requirements. 
        self.forbiddenFrequencies = [3600, 3500] # Forbidden frequenciues (eg distress beacon)
        self.fullRoleList = ['Police', 'CFS', 'Ambulance', 'Defence', 'Farmers', 'Civilians'] # CFS - Country Fire Service
        self.roleList = ['Police', 'CFS', 'Civilians']
        self.userList = [random.randint(0, len(self.roleList)-1) for temp in range(self.swarmSize)]
        self.userTensor = torch.tensor(self.userList).cuda()
        self.timeSlots = torch.torch.randint(1, 25, (3,)).cuda()
        self.geographical = ['Region1','Mountain','Region2']
          
    def visualise(self):
        # Showing Particle Graphs. 
        for graph in self.graphs:
            graph.showPlot()
            
    def initialiseGraphs(self):
        # Initialising Scatter Plot Graphs
        self.graphs = []        
        for role in self.roleList:
            self.graphs.append(ScatterPlot(role, self.maxIterations))  
        
    # Objective Function. Creates a scoring system based on the constraints of the scenario. 
    def objectiveFunction(self, frequency):
        
        # CONSTRAINT 1: Frequencies that can't be used eg distress beacons. 
        # Default deters away from frequencies 3500-3600.
        restrictedPenalty = torch.zeros((self.swarmSize), dtype=torch.float).cuda()
        restrictedMask = ((frequency >= np.log10(self.forbiddenFrequencies[1])) & (frequency <= np.log10(self.forbiddenFrequencies[0])))
        restrictedPenalty[restrictedMask] = 10**10
        
        # CONSTRANT 2: Frequency used by particular operators eg police need to talk to all other police in the area. 
        # Solution (Part 1): Proximity Evaluation. Grouped together on adjacent channels - ensures they can contact one another.  
        # Punish based on how far the particle is from the average of other particles of the same group.
        proximityPenalty = torch.zeros((self.swarmSize), dtype=torch.float).cuda()
        repulsionPenalty = torch.zeros((self.swarmSize), dtype=torch.float).cuda()
        uniqueTypes = torch.unique(self.userTensor).cuda() # The different types possible (0 Police, 1 Civilian, 2 Fire). 

        for index, userType in enumerate(uniqueTypes):
            frequenciesType = frequency[self.userTensor == userType] # Selects frequencies which are of the userType.            
            typeMean = torch.mean(frequenciesType).cuda()
            proximityPenalty[self.userTensor == userType] = torch.square(frequenciesType - typeMean).cuda() 
                                 
            # Appending to the graphs based on types. 
            for i in range(0, len(frequenciesType)):
                
                self.graphs[index].append(frequenciesType.cpu().numpy()[i], i)    
                
            # Solution (Part 2): Repulsion Evaluation. Minimises interference by ensuring other groups are elsewhere. 
            # Repels certain groups from placing near one another. 
            # Punish based on how close the particle is from the average of the other particles of different groups.           
            for otherUserType in uniqueTypes:
                if otherUserType != userType:
                    otherFrequencyType = frequency[self.userTensor == otherUserType] # Make a tensor for each group type. 
                    otherTypeMean = torch.mean(otherFrequencyType).cuda() # Calculate the mean of the group type. 
                    otherAbsDiff = torch.abs(frequenciesType - otherTypeMean).cuda() # Calculate the absolute difference between frequencies and other group mean.
                    
                    # CONSTRAINT 3: Time of day that the frequency is used - e.g. can use the same frequency at different times of day.
                    # Peak Time is define from 5pm to 12am. If in peak time, the repulsion is stronger (doubled) to further avoid interferences. 
                    if self.timeSlots[otherUserType] >= 17:
                        repulsionPenalty[self.userTensor == userType] += 2*(torch.ones(len(frequenciesType)).cuda() / (torch.ones(len(frequenciesType))).cuda() *1.0e-10 + torch.square(otherAbsDiff).cuda()) # Penalty is doubled because of peak congestion time.         
                    else:
                        repulsionPenalty[self.userTensor == userType] += (torch.ones(len(frequenciesType)).cuda() / (torch.ones(len(frequenciesType))).cuda() *1.0e-10 + torch.square(otherAbsDiff).cuda()) # Append repulsion score. PSO will seek to maximise this value.   
        
        totalPenalty = restrictedPenalty + (200*proximityPenalty) + (repulsionPenalty) # PSO tool is trying to minimise this. 
        return totalPenalty

    def search_space(self):
        self.upperBound = torch.tensor(self.frequencyBounds[0]).cuda()
        self.lowerBound = torch.tensor(self.frequencyBounds[1]).cuda()
        self.dimensionality = self.upperBound.size()    

    def populate(self):
        self.frequency = ((self.upperBound - self.lowerBound)*torch.rand(self.swarmSize).cuda()).cuda() + self.lowerBound
        self.velocity = (2*self.velocityLimit*torch.rand(self.swarmSize).cuda()).cuda() - self.velocityLimit
        
    def minVal(self, tensor):
        minVal = tensor[0]
        if len(tensor) > 0:
            for value in tensor:
                if value < minVal:
                    minVal = value
        return float(minVal.item())
    
    def getIndex(self, value, tensor):
        for index, current in enumerate(tensor):
            if current == value:
                return index
    
    def enforce_bounds(self):
        upperBound = self.upperBound.view(self.dimensionality,1)
        lowerBound = self.lowerBound.view(self.dimensionality,1)
        self.frequency = torch.max(torch.min(self.frequency,upperBound),lowerBound).cuda()
        self.velocity = torch.max(torch.min(self.velocity,torch.tensor(self.velocityLimit).cuda()),-torch.tensor(self.velocityLimit).cuda())

    def run(self,verbosity = True):
        self.currentFitness = self.objectiveFunction(self.frequency) # Evaluates the fitness function at each particle position. 
        self.particleBest = self.frequency
        self.particleBestFitness = self.currentFitness
        self.globalBest = torch.full((self.swarmSize,), 99.0).cuda() # Placeholder. High enough that it will always be replaced. 
        self.globalBestFitness = torch.full((self.swarmSize,), 99999.0).cuda() # Placeholder. High enough that it will always be replaced. 

        # Multiple different swarms - global bests/fitnesses
        for i in range(0,  len(torch.unique(self.userTensor).cuda())):
            groupMask = (self.userTensor == i)
            minFitness = self.minVal(self.particleBestFitness[groupMask])
            minIndex = self.getIndex(minFitness, self.particleBestFitness[groupMask])
            minValue = self.frequency[minIndex]
            self.globalBest.masked_fill_(groupMask, minValue)
            self.globalBestFitness.masked_fill_(groupMask, minFitness)        

        for iteration in range(self.maxIterations):
            tic = time.monotonic()
            self.velocity = ((self.inertiaWeight*self.velocity) +  
                            (self.c_cognitive*torch.rand(1).cuda()*(self.particleBest-self.frequency)) + 
                            (self.c_social*torch.rand(1).cuda()*(self.globalBest-self.frequency)))
            self.frequency = self.frequency + self.velocity
            self.enforce_bounds()
            self.currentFitness = self.objectiveFunction(self.frequency) # Penalty function. 
            localMask = self.currentFitness<self.particleBestFitness # Boolean, true when current fitness less than particle best fitness. 
            self.particleBestFitness[localMask] = self.currentFitness[localMask]
            self.particleBest[localMask] = self.frequency[localMask]
                          
            # Multiple different swarms - global bests/fitnesses
            for i in range(0, len(torch.unique(self.userTensor).cuda())):
                groupMask = (self.userTensor == i)
                minFitness =self.minVal(self.currentFitness[groupMask])
                minIndex = self.getIndex(minFitness, self.currentFitness[groupMask])
                minValue = self.frequency[minIndex]
                oldMin = self.minVal(self.globalBestFitness[groupMask])

                if minFitness < oldMin:
                    self.globalBestFitness.masked_fill_(groupMask, minFitness)
                    self.globalBest.masked_fill_(groupMask, minValue)
                                       
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f} | current mean fitness {:.3f} | iteration time {:.3f}'
                .format(iteration + 1,self.globalBestFitness.min(),self.currentFitness.mean(),toc-tic))
 
        # Printing final group frequencies and averages.  
        print("====================================PSO ALLOCATION OUTCOME====================================")        
        for i in range(0, len(torch.unique(self.userTensor).cuda())):
            print(f"Frequencies Group {i+1}: ", end='')
            groupMask = (self.userTensor == i)
            sum=0
            for index, value in enumerate(self.frequency[groupMask]):
                sum += value
                print(f"{value:3f}    ", end='')
            print()
            print(f"Group {i+1} Average: {sum/(index+1):.3f}") 
         
    def executeTool(self):
        p.initialiseGraphs()
        p.search_space()
        p.populate()
        p.run()
        p.visualise()
          
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
            plt.plot(self.data['x'][i], self.data['y'], color=self.colors[i], label=f"Particle {i+1}: {self.data['x'][i][-1]}")
        plt.ylabel('Iteration Number')
        plt.xlabel('log10 (Frequency in kHz)')
        plt.title(self.type)
        plt.grid(True)
        plt.legend()
        plt.show()
                    
    def colorCheck(self, userTensor, index):
        return self.colors[userTensor[0][index]]
        
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Prompting the user for input to configure the tool. 
    print("|===============================================================|")
    print("|Default Parameters:                                            |")
    print("|[0]  swarmSize: 10                                             |")
    print("|[1]  c_cognitive: 2                                            |")
    print("|[2]  c_social: 2                                               |")
    print("|[3]  inertiaWeight: 0.5                                        |")
    print("|[4]  velocityLimit: 0.5                                        |")
    print("|[5]  maxIterations: 100                                        |")
    print("|[6]  roles (randomly assigned): Police, CFS, Civilians.        |")
    print("|[7]  timeSlots: 1-24, one for each role allocated.             |")
    print("|[8]  geographical: Region1, Mountain, Region2.                 |")
    print("|[9]  frequencyBounds: 3000 - 30000 (HF Band - kHZ)             |")
    print("|[10] forbiddenFrequencies: 3500 - 3600                         |")
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
                    print("The list of choosable roles are: ", end='')
                    for role in p.fullRoleList:
                        print(f"{role}", end=" ")
                    print()
                    roles=[]
                    numRoles = int(input("How many roles: "))
                    for i in range(0, numRoles):
                        while len(roles) != i+1:
                            role = str(input(f"Enter role {i+1}: "))
                            if role in p.fullRoleList and role not in roles:
                                roles.append(role)
                    p.roleList = roles
                case 7:
                    times = []
                    for i in range(0, len(p.roleList)):
                        times.append(int(input(f"Enter time {i+1}: ")))
                    p.timeSlots = torch.tensor(times).cuda()
                case 8:
                    numGeo = int(input("Enter number of desired regions: "))
                    geo = []
                    for i in range(0, numGeo):
                        geo.append(input(f"Enter geographical location {i+1}: "))
                    p.geographical = geo
                case 9:
                    bounds = [0,0]
                    print("Bands (Units in kHz): \nHigh Frequency (HF) -> 3000-30000\nVery High Frequency (VHF) -> 30000-300000\nUltra High Frequency (UHF) -> 300000-3000000")
                    bounds[0] = int(input("Enter upper frequency bound: "))
                    bounds[1] = int(input("Enter lower frequency bound: "))
                    p.frequencyBounds = bounds
                case 10:
                    forbidden = [0,0]
                    forbidden[0] = int(input("Enter forbidden range (upper bound): "))
                    forbidden[1] = int(input("Enter forbidden range (lower bound): "))
                    p.forbiddenFrequencies = forbidden
   
            inputBool = str(input("Do you want to change other parameters? [y/n]: "))
    
    p.executeTool()
   