from deap import base
from deap import creator
from deap import tools
import random
import numpy
from deap.tools.crossover import cxOrdered
from deap.benchmarks.tools import hypervolume
import matplotlib.pyplot as plt
import pandas as pd

GENS = 250          #Number of generations
NUM_INDS = 40       #Number of individuals on a generations

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

#Get the cities coordinates array and remove the cities we will not consider
positions = pd.read_csv("Datasets/CitiesXY.csv")
positions = positions.drop(columns=["City"])

for i in range(num_cities, 50):
    positions = positions.drop([i])


#This function detects a stretch of 4 or more plane trips in a row and changes 
#the 4th trip to a car trip
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

#Initializes our individual randomly, taking into account our hard constraint
# (maximum of 3 plane trips in a row). Our individual contains a list of lists:
# the first list indicates the order in which the salesman visits all the cities
# and the second one the means of transportation (0 and 1 representing a car 
# and a plane trip, respectively)
def initInd(icls):
    ind = [[] for i in range(2)]
    transportes = []
    
    flag = 0
    while(flag==0):
        for i in range(num_cities):
            transportes.append(random.randint(0, 1))
        
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
        
    ind[0]=random.sample(range(num_cities), num_cities)
    ind[1]=transportes
    
    return icls(ind)

#This function receives our individual and returns the distance(time) that said route
# takes to complete
def city_distance(individual):
    data_car = pd.read_csv("Datasets/CityDistCar.csv")
    data_car = data_car.drop(columns=["Distances of Cities by Car (min)"])
    
    data_plane = pd.read_csv("Datasets/CityDistPlane.csv")
    data_plane = data_plane.drop(columns=["Distances of Cities by Flight (in min)"])
    
    if individual[1][-1] == 0:
            distance = data_car.iloc[individual[0][-1], individual[0][0]]
            
    if individual[1][-1] == 1:
            distance = data_plane.iloc[individual[0][-1], individual[0][0]]
    
    for i in range(num_cities-1):
        if individual[1][i] == 0:
            distance += data_car.iloc[individual[0][i], individual[0][i+1]]
        if individual[1][i] == 1:
            distance += data_plane.iloc[individual[0][i], individual[0][i+1]]
    
    return distance

#Acts the same as the last function but with regards to the cost of the trip
def city_cost(individual):
    data_car = pd.read_csv("Datasets/CityCostCar.csv")
    data_car = data_car.drop(columns=["Cost of Cities by Car (€)"])
    
    data_plane = pd.read_csv("Datasets/CityCostPlane.csv")
    data_plane = data_plane.drop(columns=["Cost of Cities by Flight (€)"])
    
    if individual[1][-1] == 0:
            cost = data_car.iloc[individual[0][-1], individual[0][0]]
            
    if individual[1][-1] == 1:
            cost = data_plane.iloc[individual[0][-1], individual[0][0]]
    
    for i in range(num_cities-1):
        if individual[1][i] == 0:
            cost += data_car.iloc[individual[0][i], individual[0][i+1]]
        if individual[1][i] == 1:
            cost += data_plane.iloc[individual[0][i], individual[0][i+1]]
    
    return cost

#Receives our individual and evaluates its fitness, returning a tuple containing
# the distance(time) and cost, respectively, that it takes to complete the route
def evalPopulation(individual):
    distance = city_distance(individual)
    cost = city_cost(individual)
    return distance, cost

#Our modified ordered crossover function: performs a normal ordered crossover
# on the cities route array but performs a two point crossover on our mode of 
# transport array. Also takes into account the hard constraint by running the 
# appropriate function (hard_constraint)
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

#Our modified Shuffle Indexes mutation function: performs a normal Shffle 
# indexes crossover but takes into account our hard constraint
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

#Analyses our distance and cost data files and generates the reference for the
# hypervolume function
def hypervolume_ref():
    car_dist = pd.read_csv("Datasets/CityDistCar.csv")
    car_dist = car_dist.drop(columns=["Distances of Cities by Car (min)"])
    
    plane_dist = pd.read_csv("Datasets/CityDistPlane.csv")
    plane_dist = plane_dist.drop(columns=["Distances of Cities by Flight (in min)"])
    
    car_cost = pd.read_csv("Datasets/CityCostCar.csv")
    car_cost = car_cost.drop(columns=["Cost of Cities by Car (€)"])
    
    plane_cost = pd.read_csv("Datasets/CityCostPlane.csv")
    plane_cost = plane_cost.drop(columns=["Cost of Cities by Flight (€)"])

    dist1 = car_dist.iloc[:num_cities, :num_cities].to_numpy().max() * num_cities
    dist2 = plane_dist.iloc[:num_cities, :num_cities].to_numpy().max() * num_cities
    cost1 = car_cost.iloc[:num_cities, :num_cities].to_numpy().max() * num_cities
    cost2 = plane_cost.iloc[:num_cities, :num_cities].to_numpy().max() * num_cities
    
    if dist1 > dist2:
        dist_max = dist1
    else:
        dist_max = dist2
        
    if cost1 > cost2:
        cost_max = cost1
    else:
        cost_max = cost2
        
    return (float(dist_max), float(cost_max))

#Initialization of our toolbox variables
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", initInd, icls=creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalPopulation)
toolbox.register("mate", cxOrdered_modified)
toolbox.register("mutate", mutShuffleIndexes_modified, indpb= 1/num_cities)
toolbox.register("select", tools.selNSGA2)
pop = 0

def main():
    CXPB=0.7    #Crossover probability
    
    #Randomizes our seed
    seed = random.randint(1, 1000)
    random.seed(seed)
    
    hypervolumes = []
    
    #Initializations for our pareto front functions
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    pareto = tools.ParetoFront()
    
    #Generates the population
    pop = toolbox.population(n=NUM_INDS)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, GENS):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:     #Defines when to perform crossover
                toolbox.mate(ind1, ind2)
                
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, NUM_INDS)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        pareto.update(pop)
        
        best_ind = tools.selBest(pop, 1)[0]
        
        hypervolumes.append(hypervolume(pareto, hypervolume_ref()))
        
    
    return pareto, best_ind, hypervolumes


    
if __name__ == "__main__":
    optimal_front, best_ind, hypervolumes = main()
    
    #Plot the best tour
    plt.title('Optimized tour')
    plt.scatter(positions["x"], positions["y"])

    for i in range(0, num_cities-1):
        start_pos = best_ind[0][i]
        end_pos = best_ind[0][i+1]
        if best_ind[1][i+1] == 0:
            plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='r'))
        else:
            plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='b'))

    start_pos = best_ind[0][num_cities-1]
    end_pos = best_ind[0][0]
    if best_ind[1][i+1] == 0:
        plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='r'))
    else:
        plt.annotate("", xy=(positions.iloc[end_pos]["x"], positions.iloc[end_pos]["y"]), xytext=(positions.iloc[start_pos]["x"], positions.iloc[start_pos]["y"]), arrowprops=dict(arrowstyle="->", color='b'))


    plt.tight_layout()
    plt.show()
    
    #Plot the pareto front
    y, x = zip(*[ind.fitness.values for ind in optimal_front])

    min_index_y = y.index(min(y))
    print("Lowest distance and its cost")
    print(y[min_index_y], x[min_index_y])
    
    min_index_x = x.index(min(x))
    print("Lowest cost and its distance")
    print(y[min_index_x], x[min_index_x])

    fig = plt.figure()
    fig.set_size_inches(15,10)
    
    axe = plt.subplot2grid((2,2),(0,0))
    axe.set_ylabel('Distância', fontsize=12)
    axe.set_xlabel('Custo', fontsize=12)
    axe.scatter(x, y, c='b', marker='+')
    
    plt.show()
    
    #Plot the hypervolume evolution curve
    plt.title('Hypervolume evolution curve')
    x = list(range(1, GENS))
    
    plt.plot(x, hypervolumes)
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.show()
    
