from deap import base
from deap import creator
from deap import tools
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Declaration of Constants
# POPULATION - Number of genes in population
# GENERATIONS - Number of generations
# MAX_RUNS - Number of runs with different random seeds
# CXPB - is the probability with which two individuals are crossed
# MUTPB - is the probability for mutating an individual
##
POPULATION = 40
GENERATIONS = 250
MAX_RUNS = 1
CXPB = 0.9
MUTPB = 0.3

## Terminal UI to choose between the different datasets
print("***********************************************")
print("*--- Single-Objective Optimization Problem ---*")
print("*    Choose One :                             *")
print("*         1 - Car Distances                   *")
print("*         2 - Car Costs                       *")
print("*         3 - Plane Distances                 *")
print("*         4 - Plane Costs                     *")
print("*                                             *")
print("*         0 - EXIT                            *")
print("***********************************************")
status = input()
while True:
    if status == '1':
        data = pd.read_csv("Datasets/CityDistCar.csv")
        data = data.drop(columns=["Distances of Cities by Car (min)"])
        break
    if status == '2':
        data = pd.read_csv("Datasets/CityCostCar.csv")
        data = data.drop(columns=["Cost of Cities by Car (€)"])
        break
    if status == '3':
        data = pd.read_csv("Datasets/CityDistPlane.csv")
        data = data.drop(columns=["Distances of Cities by Flight (in min)"])
        break
    if status == '4':
        data = pd.read_csv("Datasets/CityCostPlane.csv")
        data = data.drop(columns=["Cost of Cities by Flight (€)"])
        break
    if status == '0':
        quit()
    status = input()

## Terminal UI to choose between the number of cities to run
print("***********************************************")
print("*--- Single-Objective Optimization Problem ---*")
print("*    Number of Cities :                       *")
print("*         1 - 20                              *")
print("*         2 - 30                              *")
print("*         3 - 50                              *")
print("***********************************************")
num_cities = 20
mode = input()
while True :
    if mode == '1':
        num_cities = 20
        break
    if mode == '2':
        num_cities = 30
        break
    if mode == '3':
        num_cities = 50
        break
    print("Entry not Valid, try again!")
    mode = input()

## Terminal UI to choose if it is to use with Heuristics or not
print("***********************************************")
print("*--- Single-Objective Optimization Problem ---*")
print("*    Choose One :                             *")
print("*         1 - Normal                          *")
print("*         2 - Heuristic                       *")
print("***********************************************")
mode = input()
while True :
    if mode == '1':
        break
    if mode == '2':
        break
    print("Entry not Valid, try again!")
    mode = input()

## Read the dataset CitiesXY.csv and store it in positions variable 
positions = pd.read_csv("Datasets/CitiesXY.csv")
positions = positions.drop(columns=["City"])
## Filter the number of cities to be used with the Algorithm
positions.drop(positions.tail(50 - num_cities).index,inplace=True) # drop last n rows

## Heuristic function
#
#
def heuristic():
    
    positions = pd.read_csv("Datasets/CitiesXY.csv")
    positions = positions.drop(columns=["City"])

    for i in range(num_cities, 50):
        positions = positions.drop([i])
    
    data_left = positions.where(positions['x']<500)
    data_right = positions.where(positions['x']>500)
    
    data_left = data_left.dropna()
    data_right = data_right.dropna()
    
    data_left = data_left.sort_values(by='y')
    data_right = data_right.sort_values(by='y', ascending=False)
    
    data_heur = data_left.append(data_right)
    
    heuristic = data_heur.index.values.tolist()
            
    return heuristic

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("rand_city", random.sample, range(num_cities), num_cities)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.rand_city)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def city_distance(individual):
    distance = 0
    distance = data.iloc[individual[0], individual[-1]]
    
    for i in range(num_cities - 1):
        distance += data.iloc[individual[i], individual[i+1]]
    
    return distance,


toolbox.register("evaluate", city_distance)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb= 1/num_cities)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():

    overall = []
    seed = 40
    shortest = 100000

    for rnd in range(0, MAX_RUNS):
        random.seed(9)

        print("-- RUN %i --" % rnd)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=POPULATION)

        if(mode == '2'):
            pop[POPULATION-1]=creator.Individual(heuristic())

        
        print("Start of evolution")
        
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of 
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        
        bestGen = []
        
        
        # Begin the evolution
        while g < GENERATIONS:

            # A new generation
            g = g + 1
            print("-- Generation %i, Run %i --" % (g, rnd))

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            
            best_ind = tools.selBest(pop, 1)[0]
            print("Gen %d: best individual is %s, %s" % (g, best_ind, best_ind.fitness.values))
            bestGen.append(best_ind.fitness.values)

            if shortest > best_ind.fitness.values[0]:
                seed = rnd
                shortest = best_ind.fitness.values[0]
                

        
        print("-- End of (successful) evolution --")
        
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        overall.append(best_ind.fitness.values[0])


    print("MEAN = %i STD = %i" % (np.mean(overall), np.std(overall)))
    print("SEED %i %i" % (seed, shortest))
    
    plt.title('Optimized tour')

    plt.scatter(positions["x"], positions["y"])

    for i in range(0,num_cities-1):
        start_pos = best_ind[i]
        end_pos = best_ind[i+1]
        plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='r'))
    
    start_pos = best_ind[num_cities-1]
    end_pos = best_ind[0]
    plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='b'))

    # textstr = "N nodes: %d\nTotal length: %s" % (num_cities, best_ind.fitness.values)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.text(0.05, 0.95, textstr, transform=plt.transAxes, fontsize=14, # Textbox
    #         verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    plt.title('Minimum Evolution')
    plt.xlabel('Number of Generations')
    plt.ylabel('Minimum Cost in euros')
    plt.plot(bestGen)
    plt.show()

if __name__ == "__main__":
    main()
