#!/usr/bin/env python

# Library Imports
from enum import unique
from importlib.readers import FileReader
from threading import local
from weakref import proxy
from xml.dom import UserDataHandler
import torch
import time
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import cycle

# Main Class for Particle Swarm Optimization
class ParticleSwarmOptimizer(object):
    def __init__(self,swarm_size=100,options=None):
        if (options == None):
            options = [2,2,0.1,0.1,100]
        self.swarm_size = swarm_size
        self.c_cognitive = options[0] # Amount the particle is influenced by the personal best. 
        self.c_social = options[1] # Amount the particle is influenced by the global best.
        self.inertia_weight = options[2] # Influence of the current velocity on its future velocity. 
        self.velocity_limit = options[3] # How fast the particles move within the space. 
        self.max_iterations = options[4] # How many times the code runs. 

    def optimize(self,function):
        self.fitness_function = function

    def search_space(self,upper_bound,lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.dimensionality = upper_bound.size()[0]    

    def populate(self):
        self.frequency = ((self.upper_bound - self.lower_bound)*torch.rand(1,self.swarm_size)) + self.lower_bound
        print("starting: ", self.frequency)
        self.velocity = (2*self.velocity_limit*torch.rand(self.dimensionality,self.swarm_size)) - self.velocity_limit
        print("starting2: ", self.velocity)

    def enforce_bounds(self):
        upper_bound = self.upper_bound.view(self.dimensionality,1)
        lower_bound = self.lower_bound.view(self.dimensionality,1)
        self.frequency = torch.max(torch.min(self.frequency,upper_bound),lower_bound)
        self.velocity = torch.max(torch.min(self.velocity,torch.tensor(self.velocity_limit)),-torch.tensor(self.velocity_limit))

    def run(self,verbosity = True):
        self.current_fitness = self.fitness_function(self.frequency) # Evaluates the fitness function at each particle position. 
        self.particle_best = self.frequency
        
        self.particle_best_fitness = self.current_fitness
        self.global_best = self.frequency[:,self.particle_best_fitness.argmin()].view(self.dimensionality,1)  # Picks particle with the lowest penalty. 
        print("Inital Global Best: ", self.global_best)
        self.global_best_fitness = self.particle_best_fitness.min()
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            
            self.velocity = ((self.inertia_weight*self.velocity) +  # Current velocity times some weight
                            (self.c_cognitive*torch.rand(1)*(self.particle_best-self.frequency)) + # plus how far it is from its personal best times a random number. 
                            (self.c_social*torch.rand(1)*(self.global_best-self.frequency)))
            print("Velocity: ", self.velocity)
            self.frequency = self.frequency + self.velocity
            self.enforce_bounds()
            self.current_fitness = self.fitness_function(self.frequency) # Penalty function. 
            local_mask = self.current_fitness<self.particle_best_fitness # Boolean, true when current fitness less than particle best fitness. 
            self.particle_best[local_mask] = self.frequency[local_mask] # Updates the personal best positions for those only who were true. 
            self.particle_best_fitness[local_mask] = self.current_fitness[local_mask] # Updates particle best fitness values for those who were true. 
            
            # If the fitness function is less than the global best fitness, update the global best fitness. 
            if (self.current_fitness.min() < self.global_best_fitness):
                self.global_best_fitness = self.current_fitness.min()
                self.global_best = self.frequency[:,self.current_fitness.argmin()].view(self.dimensionality,1)
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f} | current mean fitness {:.3f} | iteration time {:.3f}'
                .format(iteration + 1,self.global_best_fitness,self.current_fitness.mean(),toc-tic))
                
class ScatterPlot:
    def __init__(self):
        self.data = {'x': [], 'y': []}
        self.colors = ['red','orange','yellow','green','blue','purple','pink']
            
    def append(self, x, y):
        self.data['x'].append(x)
        self.data['y'].append(y)
            
    def showPlot(self, title, color):
        plt.plot(self.data['x'], self.data['y'], color=color)
        plt.xlabel('Frequency')
        plt.ylabel('Iteration')
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    def colorCheck(self, userTensor, index):
        return self.colors[userTensor[0][index]]
        
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    SwarmSize = 10
    iterations = 100
    
    # Initialising Scatter Plot Graphs
    graphs = []        
    iterationCount = [0]
    for i in range(1, SwarmSize+1):
        graphs.append(ScatterPlot())
        
    # Bounds -  Will be the frequency limits. 
    UpperBound = torch.tensor([2000])
    LowerBound = torch.tensor([0])
    
    # Time Intervals

    #  Creates a tensor of the types each particle are 
    roleList = ['Police', 'Civilian', 'Fire'] # For Readability: If 0, Police. If 1, Civilian. If 2, Fire. 
    userList = [[random.randint(0, 2) for _ in range(SwarmSize)]]
    userTensor = torch.tensor(userList)
    plotCount = 0

    # Objective Function. Creates a scoring system based on the constraints of the scenario. 
    def ObjectiveFunction(frequency):
        
        # Adding to Scatter Plot.
        iterationCount[0] += 1
        print("Frequency: ", frequency)
        for index in range(0, SwarmSize):
            graphs[index].append(frequency.numpy()[0][index], iterationCount[0])
   
        # Target Value: 500. Incentivises Frequencies to converge towards this point. Parabolic Function. 
        target = torch.full((1,SwarmSize), 500.0)
        targetPenalty = torch.square(frequency - target)
        print("Target Penalty: ", targetPenalty)

        # Detering away from 550-600.
        restrictedPenalty = torch.zeros((1, SwarmSize))
        restrictedMask = (frequency >= 550) & (frequency <= 650)
        restrictedPenalty[restrictedMask] = 10**10
        print("Restricted Penalty: ", restrictedPenalty)
        
        # Proximity Evaluation. 
        # Punish based on how far the particle is from the average of other particles of the same group.
        proximityPenalty = torch.zeros((1, SwarmSize))
        uniqueTypes = torch.unique(userTensor) # The different types possible (0 Police, 1 Civilian, 2 Fire). 
        for userType in uniqueTypes:
            frequenciesType = frequency[userTensor == userType] # Selects frequencies which are of the userType.            
            distCentre = torch.mean(frequency[userTensor == userType]) 
            proximityPenalty[userTensor == userType] = torch.sum(torch.abs(frequenciesType - distCentre)) 
                                 
            # Print Statements
            print("User Tensor: ", userTensor)
            print("Frequencies of the Current Type: ", userType, " ", frequenciesType)
            print("Centre of Distribution: ", distCentre)
            print("Proximity Penalty: ", proximityPenalty)
            
        # Repulsion Evaluation. WIP
        # Repels certain groups from converging on top of one another. 
                    # PROXIMITY CHECK - REPELLING DIFFERENT TYPES
            # Builds off the grouping similar types section. 
            
            for otherUserType in uniqueTypes:
                if otherUserType != userType:
                    otherFrequenciesType = frequency[userTensor == otherUserType]
                    if len(frequenciesType > 1): # Verifying that there's more than one frequency.
                        repulsion += torch.mean(torch.abs(frequenciesType - torch.mean(otherFrequenciesType)))
                        print("Repulsion: ", repulsion)
            
            cost = similarity - repulsion
            print("Cost function: ", cost) # The higher this is the better. 
                
         
            

        totalPenalty = targetPenalty + restrictedPenalty + (10*proximityPenalty) # It is trying to minimise this. 
        print("Total Penalty: ", totalPenalty)
        return totalPenalty

    p = ParticleSwarmOptimizer(SwarmSize, options=[2,2,0.1,10,iterations]) # Options order: Cognitive, Social, Inertia, Velocity Limit, Iterations. 
    p.optimize(ObjectiveFunction)
    p.search_space(UpperBound,LowerBound)
    p.populate()
    p.run()
    
    # Showing Particle Graphs. 
    #cycler = cycle(graphs[0].colors)
    for i in range(1, SwarmSize+1):
        graphs[i-1].showPlot("Spectrum Allocation: Swarm Particle {}".format(i), color=graphs[0].colorCheck(userTensor,i-1)) #color=next(cycler))
    plt.show()