#!/usr/bin/env python

# Library Imports
from inspect import Parameter
from unittest.mock import DEFAULT
import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Main Class for Particle Swarm Optimization
class ParticleSwarmOptimizer(object):
    def __init__(self):
        self.swarmSize = 10
        self.c_cognitive = 10 # Amount the particle is influenced by the personal best. 
        self.c_social = 10 # Amount the particle is influenced by the global best. #DEFault 2
        self.inertiaWeight = 2 # Influence of the current velocity on its future velocity. 
        self.velocityLimit = 2 # How fast the particles move within the space. 
        self.maxIterations = 100 # How many times the code runs. 
        self.frequencyBounds = [np.log10(30000), np.log10(3000)] # Upper and lower frequency bounds. Default is the HF band.  
        
        # Addressing the requirements. 
        self.forbiddenFrequencies = [3600, 3500] # Forbidden frequenciues (eg distress beacon)
        self.fullRoleList = ['Police', 'CFS', 'Ambulance', 'Defence', 'Farmers', 'Civilians'] # CFS - Country Fire Service
        self.roleList = ['Police', 'CFS', 'Civilians']
        self.userList = [random.randint(0, len(self.roleList)-1) for temp in range(self.swarmSize)]
        self.userTensor = torch.tensor(self.userList)
        self.timeSlots = torch.linspace(0, 24, steps=self.swarmSize)
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
        # Default deters away from frequencies 550-650.
        restrictedPenalty = torch.zeros((self.swarmSize), dtype=torch.float)
        restrictedMask = ((frequency >= self.forbiddenFrequencies[1]) & (frequency <= self.forbiddenFrequencies[0]))
        restrictedPenalty[restrictedMask] = 10**10
        print("Restricted Penalty: ", restrictedPenalty)
        
        # CONSTRANT 2: Frequency used by particular operators eg police need to talk to all other police in the area. 
        # Solution (Part 1): Proximity Evaluation. Grouped together on adjacent channels - ensures they can contact one another.  
        # Punish based on how far the particle is from the average of other particles of the same group.
        proximityPenalty = torch.zeros((self.swarmSize), dtype=torch.float)
        repulsionReward = torch.zeros((self.swarmSize), dtype=torch.float)
        uniqueTypes = torch.unique(self.userTensor) # The different types possible (0 Police, 1 Civilian, 2 Fire). 

        for index, userType in enumerate(uniqueTypes):
            frequenciesType = frequency[self.userTensor == userType] # Selects frequencies which are of the userType.            
            typeMean = torch.mean(frequenciesType)
            proximityPenalty[self.userTensor == userType] = torch.square(frequenciesType - typeMean) 
                                 
            # Print Statements
            print("User Tensor: ", self.userTensor)
            print("Frequencies of the Current Type: ", userType, " ", frequenciesType)
            print("Sim Mean: ", typeMean)
            print("Proximity Penalty current Type: ", torch.mul(proximityPenalty[self.userTensor == userType], 200))
            print("tensor: ", torch.ones(1, self.swarmSize))
            # Appending to the graphs based on types. 
            for i in range(0, len(frequenciesType)):
                self.graphs[index].append(frequenciesType.numpy()[i], i)    
                
            # Solution (Part 2): Repulsion Evaluation. Minimises interference by ensuring other groups are elsewhere. 
            # Repels certain groups from placing near one another. 
            # Punish based on how close the particle is from the average of the other particles of different groups.           
            for otherUserType in uniqueTypes:
                if otherUserType != userType:
                    otherFrequencyType = frequency[self.userTensor == otherUserType] # Make a tensor for each group type. 
                    otherTypeMean = torch.mean(otherFrequencyType) # Calculate the mean of the group type. 
                    otherAbsDiff = torch.abs(frequenciesType - otherTypeMean) # Calculate the absolute difference between frequencies and other group mean.
                    repulsionReward[self.userTensor == userType] += torch.ones(len(frequenciesType)) / (torch.ones(len(frequenciesType)) + torch.square(otherAbsDiff)) # Append repulsion score. PSO will seek to maximise this value. 
                    
                    print("Diff mean: ", otherTypeMean)
            print("Repulsion Penalty: ", repulsionReward[self.userTensor == userType])        
            

        # CONSTRAINT 3: Frequency Capabilties of the radio equipment 
        # Interpretation: What frequencies can certain users use? What are the limitations?
        # Possible Users: Police, CFS, Ambulance, Defence, Farmers, Civilians.
        #   Police: Two-way radios, mobile phones, satellites. 
        #   Country Fire Service: Two-way radios, mobile phones, satellites. 
        #   Ambulance: Two-way radios, mobile phones, satellites. 
        #   Defence: Two-way radios, mobile phones, satellites. 
        #   Farmers: Two-way radios. 
        #   Civilians: Mobile Phones. 
        # Equipment Frequencies:    
        #   Two-way radios: VHF, UHF.
        #   Mobile Phones:      
        #   Ambulances: Equipment - Two-way radios, satellite phones (UHF: 1.6GHz to 2.4GHz). Satellite - global coverage, less susceptible to terrain/atmospheric conditions.     
        # FIRE: HF/VHF/UHF (All 3)
        # EMERGENCY SERVICES: HF/VHF/UHF (All 3)
        # POLICE: HF/UHF (Cannot use VHF)
        # 
        # 
        #    
        
        # HF is more reliable at night. 
        # Mountains can obstruct HF signals. 
        


        totalPenalty = restrictedPenalty + (200*proximityPenalty) + (repulsionReward) # It is trying to minimise this. 
        print("Total Penalty: ", totalPenalty)
        return totalPenalty

    def search_space(self):
        self.upperBound = torch.tensor(self.frequencyBounds[0])
        self.lowerBound = torch.tensor(self.frequencyBounds[1])
        self.dimensionality = self.upperBound.size()    

    def populate(self):
        self.frequency = ((self.upperBound - self.lowerBound)*torch.rand(self.swarmSize)) + self.lowerBound
        print("starting: ", self.frequency)
        self.velocity = (2*self.velocityLimit*torch.rand(self.swarmSize)) - self.velocityLimit
        
    def minVal(self, tensor):
        if len(tensor) > 0:
            
            minVal = tensor[0]
            
            for value in tensor:
                if value < minVal:
                    minVal = value
            print("function min: ", minVal)
            return float(minVal.item())
        return 50 # Placeholder
    
    def getIndex(self, value, tensor):
        for index, current in enumerate(tensor):
            if current == value:
                return index
    
    def enforce_bounds(self):
        upperBound = self.upperBound.view(self.dimensionality,1)
        lowerBound = self.lowerBound.view(self.dimensionality,1)
        self.frequency = torch.max(torch.min(self.frequency,upperBound),lowerBound)
        self.velocity = torch.max(torch.min(self.velocity,torch.tensor(self.velocityLimit)),-torch.tensor(self.velocityLimit))

    def run(self,verbosity = True):
        self.currentFitness = self.objectiveFunction(self.frequency) # Evaluates the fitness function at each particle position. 
        self.particleBest = self.frequency
        self.particleBestFitness = self.currentFitness
        self.globalBest = self.frequency
        self.globalBestFitness = self.frequency

        for i in range(0,  len(torch.unique(self.userTensor))):
            groupMask = (self.userTensor == i)
            minFitness = self.minVal(self.particleBestFitness[groupMask])
            minIndex = self.getIndex(minFitness, self.particleBestFitness[groupMask])
            minValue = self.frequency[minIndex]
            self.globalBest.masked_fill_(groupMask, minValue)
            self.globalBestFitness.masked_fill_(groupMask, minFitness)
        print("!!!Frequencies: ", self.frequency)
        print("!!!Penalties: ", self.currentFitness)
        print("!!!Global Best: ", self.globalBest)
        print("!!!Global Best Fitness: ", self.globalBestFitness)
        

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
            self.particleBestFitness[localMask] = self.currentFitness[localMask]
            self.particleBest[localMask] = self.frequency[localMask] # Seems fine. 
                          
            
            for i in range(0, len(torch.unique(self.userTensor))):
                groupMask = (self.userTensor == i)
                minFitness =self.minVal(self.currentFitness[groupMask])
                minIndex = self.getIndex(minFitness, self.currentFitness[groupMask])
                minValue = self.frequency[minIndex]
                oldMin = self.minVal(self.globalBestFitness[groupMask])
                print("Old Min: ", oldMin)
                print("Fitness min: ", minFitness)
                print("Fitness array: ", self.currentFitness[groupMask])
                #print("Dimension: ", tempFrequency[self.currentFitness[mask].argmin()].view(self.dimensionality, 1))
                if minFitness < oldMin:
                    self.globalBestFitness.masked_fill_(groupMask, minFitness)
                    self.globalBest.masked_fill_(groupMask, minValue)
                    


                   
            print("!!!Frequencies: ", self.frequency)
            print("!!!Penalties: ", self.currentFitness)
            print("!!!Global Best: ", self.globalBest)
            print("!!!Global Best Fitness: ", self.globalBestFitness)


                #if self.currentFitness[mask].min() < self.globalBestFitness[mask]:
                   # self.globalBestFitness[mask] = torch.tensor([self.currentFitness.min()]self.currentFitness[mask]
                    #self.globalBest[mask] = temp[self.currentFitness.argmin()].view(self.dimensionality, 1)
                    
            

           #if (self.currentFitness.min() < self.globalBestFitness):
           #     self.globalBestFitness = self.currentFitness.min()
           #     self.globalBest = self.frequency[self.currentFitness.argmin()].view(self.dimensionality,1)
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f} | current mean fitness {:.3f} | iteration time {:.3f}'
                .format(iteration + 1,self.globalBestFitness.min(),self.currentFitness.mean(),toc-tic))
                print("Final Frequency: ", self.frequency)
                
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
    print("|[3]  inertiaWeight: 0.1                                        |")
    print("|[4]  velocityLimit: 0.1                                        |")
    print("|[5]  maxIterations: 100                                        |")
    print("|[6]  roles (randomly assigned): Police, CFS, Civilians.        |")
    print("|[7]  timeSlots: 0-24, <swarm size> steps.                      |")
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