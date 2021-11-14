from deap import base
from deap import creator
from deap import tools
import random
import numpy
import json
from deap.tools.crossover import cxOrdered
import matplotlib.pyplot as plt
import pandas as pd

GENS = 40
CITIES = 20
NUM_INDS = 250

positions = pd.read_csv("CitiesXY.csv")
positions = positions.drop(columns=["City"])

for i in range(CITIES, 50):
    positions = positions.drop([i])

def hard_constraint(ind, size):
    counter = 0
    for transport in range(size):
        if ind[transport] == 1:
            counter += 1
        else:
            counter = 0
            continue
        if counter > 3:
            ind[transport] = 0
            counter = 0
    return ind


def initInd(icls):
    ind = [[] for i in range(2)]
    transportes = []
    
    flag = 0
    while(flag==0):
        for i in range(CITIES):
            transportes.append(random.randint(0, 1))
        
        # print(transportes)
        counter = 0
        for transport in transportes:
            if transport == 1:
                counter += 1
            else:
                counter = 0
            if counter >= 3:
                transportes = []
                flag = 0
                break
            else:
                flag = 1 
        
    ind[0]=random.sample(range(CITIES), CITIES)
    ind[1]=transportes
    
    return icls(ind)

def city_distance(individual):
    data_car = pd.read_csv("CityDistCar.csv")
    data_car = data_car.drop(columns=["Distances of Cities by Car (min)"])
    
    data_plane = pd.read_csv("CityDistPlane.csv")
    data_plane = data_plane.drop(columns=["Distances of Cities by Flight (in min)"])
    
    if individual[1][-1] == 0:
            distance = data_car.iloc[individual[0][-1], individual[0][0]]
            
    if individual[1][-1] == 1:
            distance = data_plane.iloc[individual[0][-1], individual[0][0]]
    
    for i in range(CITIES-1):
        if individual[1][i] == 0:
            distance += data_car.iloc[individual[0][i], individual[0][i+1]]
        if individual[1][i] == 1:
            distance += data_plane.iloc[individual[0][i], individual[0][i+1]]
    
    return distance

def city_cost(individual):
    data_car = pd.read_csv("CityCostCar.csv")
    data_car = data_car.drop(columns=["Cost of Cities by Car (€)"])
    
    data_plane = pd.read_csv("CityCostPlane.csv")
    data_plane = data_plane.drop(columns=["Cost of Cities by Flight (€)"])
    
    if individual[1][-1] == 0:
            cost = data_car.iloc[individual[0][-1], individual[0][0]]
            
    if individual[1][-1] == 1:
            cost = data_plane.iloc[individual[0][-1], individual[0][0]]
    
    for i in range(CITIES-1):
        if individual[1][i] == 0:
            cost += data_car.iloc[individual[0][i], individual[0][i+1]]
        if individual[1][i] == 1:
            cost += data_plane.iloc[individual[0][i], individual[0][i+1]]
    
    return cost

def evalPopulation(individual):
    distance = city_distance(individual)
    cost = city_cost(individual)
    return distance, cost

def cxOrdered_modified(ind1, ind2):
    size = min(len(ind1[0]), len(ind2[0]))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[0][i]] = False
            holes2[ind1[0][i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1_c, temp2_c = ind1[0], ind2[0]

    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1_c[(i + b + 1) % size]]:
            ind1[0][k1 % size] = temp1_c[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2_c[(i + b + 1) % size]]:
            ind2[0][k2 % size] = temp2_c[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[0][i], ind2[0][i] = ind2[0][i], ind1[0][i]
    


    size = min(len(ind1[1]), len(ind2[1]))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[1][cxpoint1:cxpoint2], ind2[1][cxpoint1:cxpoint2] \
        = ind2[1][cxpoint1:cxpoint2], ind1[1][cxpoint1:cxpoint2]

    hard_constraint(ind1[1], size)
    hard_constraint(ind2[1], size)

    return ind1, ind2

def mutShuffleIndexes_modified(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
    probability of each attribute to be moved. Usually this mutation is applied on
    vector of indices.
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be exchanged to
                  another position.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    """
    size = len(individual[0])
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[0][i], individual[0][swap_indx] = \
                individual[0][swap_indx], individual[0][i]
            individual[1][i], individual[1][swap_indx] = \
                individual[1][swap_indx], individual[1][i]

    hard_constraint(individual[1], size)

    return individual,


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("individual", initInd, icls=creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalPopulation)
toolbox.register("mate", cxOrdered_modified)
toolbox.register("mutate", mutShuffleIndexes_modified, indpb= 1/CITIES)
toolbox.register("select", tools.selNSGA2)
pop = 0
def main():
    random.seed(random.randint(1, 1000))
    # random.seed(1)
    
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=NUM_INDS)
    # print(pop)
    
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.9, 0.1
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits_dist = [ind.fitness.values[0] for ind in pop]
    fits_cost = [ind.fitness.values[1] for ind in pop]
    
    print(fits_dist)
    print(fits_cost)
    
    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < GENS:
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
        # resultados = np.append(resultados, best_ind)
        # print(resultados)
        print("Gen %d: best individual is %s, %s" % (g, best_ind, best_ind.fitness.values))
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))



    plt.title('Optimized tour')
    plt.scatter(positions["x"], positions["y"])

    for i in range(0,CITIES-1):
        start_pos = best_ind[0][i]
        end_pos = best_ind[0][i+1]
        if best_ind[1][i+1] == 0:
            plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='r'))
        else:
            plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='b'))

    
    start_pos = best_ind[0][CITIES-1]
    end_pos = best_ind[0][0]
    if best_ind[1][i+1] == 0:
        plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='r'))
    else:
        plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='b'))

    # textstr = "N nodes: %d\nTotal length: %s" % (num_cities, best_ind.fitness.values)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.text(0.05, 0.95, textstr, transform=plt.transAxes, fontsize=14, # Textbox
    #         verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    main()
