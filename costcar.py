from deap import base
from deap import creator
from deap import tools
import random
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Datasets/CityCostCar.csv")
data = data.drop(columns=["Cost of Cities by Car (€)"])

positions = pd.read_csv("Datasets/CitiesXY.csv")
positions = positions.drop(columns=["City"])

NUM_CITIES = 20
POPULATION = 40
GENERATIONS = 250

positions.drop(positions.tail(50 - NUM_CITIES).index,inplace=True) # drop last n rows

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("rand_city", random.sample, range(NUM_CITIES), NUM_CITIES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.rand_city)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def city_cost(individual):
    cost = 0
    cost = data.iloc[individual[0], individual[-1]]
    
    for i in range(NUM_CITIES - 1):
        cost += data.iloc[individual[i], individual[i+1]]
    
    return cost,


toolbox.register("evaluate", city_cost)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/NUM_CITIES)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    # random.seed(64)


    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=POPULATION)
    
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.9, 0.25
    
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
    
    # Begin the evolution
    while g < GENERATIONS:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
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
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    plt.title('Optimized tour')

    plt.scatter(positions["x"], positions["y"])

    for i in range(0,NUM_CITIES-1):
        start_pos = best_ind[i]
        end_pos = best_ind[i+1]
        plt.annotate("", xy=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), xytext=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), arrowprops=dict(arrowstyle="->"))
    
    start_pos = best_ind[NUM_CITIES-1]
    end_pos = best_ind[0]
    plt.annotate("", xy=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), xytext=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), arrowprops=dict(arrowstyle="->"))

    # textstr = "N nodes: %d\nTotal length: %s" % (NUM_CITIES, best_ind.fitness.values)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.text(0.05, 0.95, textstr, transform=plt.transAxes, fontsize=14, # Textbox
    #         verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
