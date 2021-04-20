import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import __init__

from code.simulated_annealing.complex_simulated_annealing import Coordinate

global LAMBDA
LAMBDA = 1


class Node:
    def __init__(self, id, coords):
        self.coords = coords
        self.id = id


'''
class City(Node):
    def __init__(self, id, x=None, y=None):
        Node.__init__(self, id)
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
'''


class FullNodeSet:
    def __init__(self, node_list, satisfaction_list, dist_matrix=None):
        self.node_list = node_list
        self.node_map = {node.id: node for node in node_list}
        self.sat_list = satisfaction_list
        self.sat_map = {node_list[i].id: satisfaction_list[i]
                        for i in range(len(node_list))}
        self.dist_matrix = dist_matrix

    def get_node(self, id):
        return self.node_map[id]


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population, node_set, cost_fn, velocity):
    fitnessResults = {}
    for i in range(0, len(population)):
        data = [[node.id for node in population[i]], node_set.sat_list,
                [node.coords for node in node_set.node_list], velocity]
        fitnessResults[i] = cost_fn(data)
    return sorted(fitnessResults.items(), key=operator.itemgetter(1),
                  reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2, node_set):
    child = []
    childP1 = []
    childP2 = []

    # print(parent1)

    bisec = int(random.random() * len(parent1))
    # print(bisec)

    childP1 = [parent1[i] for i in range(len(parent1)) if i < bisec]
    # print(childP1)
    tempP2 = [item for item in parent2[::-1] if item not in childP1]
    r_len = min(int(len(parent1) - bisec), len(tempP2))
    for i in range(r_len):
        childP2.append(tempP2[i])
    childP2 = list(np.zeros(len(parent1) - len(childP2))) + childP2[::-1]

    child = childP1 + childP2[len(childP1):]
    child_temp = list(child)

    mesh = np.array([1 if nd == 0 else 0 for nd in child])
    if 1 in mesh:
        rem_nodes = [
            nod for nod in node_set.node_list if nod not in childP1 and
            nod not in childP2]
        np.random.shuffle(rem_nodes)
        for i in range(len(child)):
            if child[i] == 0:
                new_nd = rem_nodes[-1]
                rem_nodes.pop()
                child[i] = new_nd
    if 0 in child:
        mesh2 = np.array([1 if nd == 0 else 0 for nd in child])
        print('NOOOOOOOOO', child, child_temp, mesh, mesh2)

    return child


def breedPopulation(matingpool, eliteSize, node_set, constraint_fn):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    cnt = 0
    while cnt < length:
        # print(pool)
        child = breed(pool[i], pool[len(matingpool) - i - 1], node_set)
        if constraint_fn(child):
            cnt += 1
            children.append(child)
    return children


def size_change_mut(individual, node_set, mutationRate):
    if random.random() < mutationRate:
        chance = (random.random() < 0.5)
        if chance and len(individual) < len(node_set.node_list):
            rem_nds = [
                nod for nod in node_set.node_list if nod not in individual]
            new_node = random.choice(rem_nds)
            new_pos = random.choice(range(len(individual) + 1))
            individual = individual[:new_pos] + \
                [new_node] + individual[new_pos:]
        elif not chance and len(individual) > 2:
            new_pos = random.choice(range(len(individual)))
            individual.pop(new_pos)


def swap_mut(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1


def mutate(individual, node_set, swap_mutationRate, size_mutationRate):
    if random.random() < 0.5:
        swap_mut(individual, swap_mutationRate)
    else:
        size_change_mut(individual, node_set, size_mutationRate)

    return individual


def mutatePopulation(population, node_set, constraint_fn, swap_mutationRate,
                     size_mutationRate):
    mutatedPop = []
    cnt = 0

    while cnt < len(population):
        mutatedInd = mutate(
            population[cnt], node_set, swap_mutationRate, size_mutationRate)
        if constraint_fn(mutatedInd):
            mutatedPop.append(mutatedInd)
            cnt += 1

    return mutatedPop


def nextGeneration(currentGen, eliteSize, node_set, cost_fn, constraint_fn,
                   swap_mutationRate, size_mutationRate, velocity):
    popRanked = rankRoutes(currentGen, node_set, cost_fn, velocity)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, node_set, constraint_fn)
    nextGeneration = mutatePopulation(
        children, node_set, constraint_fn, swap_mutationRate,
        size_mutationRate)
    return nextGeneration


def geneticAlgorithm(init_population, popSize, eliteSize, node_set, cost_fn,
                     constraint_fn, swap_mutationRate, size_mutationRate,
                     velocity, generations, plotting=True):
    if init_population is None:
        pop = initialPopulation(popSize, node_set.node_list)
    else:
        pop = init_population
    progress = []
    init_solns = [nod.id for nod in pop[rankRoutes(
        pop, node_set, cost_fn, velocity)[0][0]]]

    for i in range(0, generations):
        if i % 10 == 0:
            bestRouteIndex, sats = rankRoutes(
                pop, node_set, cost_fn, velocity)[0]
            print('Iter ', i, ' Sat ', sats)
        pop = nextGeneration(pop, eliteSize, node_set, cost_fn,
                             constraint_fn, swap_mutationRate,
                             size_mutationRate, velocity)
        if plotting is True:
            sat = rankRoutes(pop, node_set, cost_fn, velocity)[0][1]
            progress.append(sat)

    if plotting is True:
        plt.figure()
        plt.plot(progress)
        plt.ylabel('Satisfaction')
        plt.xlabel('Generation')
        plt.savefig('plot.png')

    bestRouteIndex, sats = rankRoutes(pop, node_set, cost_fn, velocity)[0]
    print(sats)
    final_solns = [nod.id for nod in pop[bestRouteIndex]]
    bestRoute = pop[bestRouteIndex]
    init_coords = [nod.coords for nod in node_set.node_list]
    Coordinate.plot_solution(Coordinate.satisfaction, init_coords, init_solns,
                             len(init_solns), final_solns, len(final_solns),
                             node_set.sat_list, velocity, '', save=True)
    return bestRoute


def cr_const_fn(velocity, T_max, node_set):
    def fn(nodes):
        return Coordinate.constraints([[nod.id for nod in nodes], [
            nod.coords for nod in node_set.node_list], velocity, T_max])
    return fn


'''
def generate_initial_solns(no_of_solns, node_set):
    population = []
    feasible_solution = Coordinate.get_feasible_solution(
        n, velocity, T_max, seed=None, low=0, high=11
    )
    [node_set.get_node(initial_solution[i]) for i in range(loc_bar)]
'''


def generate_initial_solns(sample, no_of_solns, node_set, constraint_fn):
    solns = []
    solns.append(sample)
    common = list(node_set.node_list)
    while len(solns) < no_of_solns:
        loc_bar = np.random.randint(1, len(common))
        np.random.shuffle(common)
        soln = common[:loc_bar]
        print(len(soln))
        if constraint_fn(soln):
            solns.append(soln)
    return solns


'''
def geneticAlgorithmPlot(population, popSize, eliteSize, node_set,
     swap_mutationRate, size_mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
'''


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    n = 40
    velocity = 1
    T_max = 20
    seedee = 1
    popsize = 100

    feasible_solution = Coordinate.get_feasible_solution(
        n, velocity, T_max, seed=seedee, low=0, high=11
    )
    initial_coords, initial_solution, S, loc_bar = feasible_solution

    node_set = FullNodeSet([Node(i, initial_coords[i])
                            for i in range(len(initial_coords))], S)

    cost_func = Coordinate.satisfaction
    constraint_func = cr_const_fn(velocity, T_max, node_set)

    population = generate_initial_solns([node_set.get_node(
        initial_solution[i]) for i in range(loc_bar)], popsize,
        node_set, constraint_func)
    print('hii', len(population))

    """ print([(initial_solution[i]) for i in range(loc_bar)])
    print([population[-1][i].id for i in range(loc_bar)]) """

    best_route = geneticAlgorithm(init_population=population,
                                  popSize=popsize, eliteSize=20,
                                  node_set=node_set, cost_fn=cost_func,
                                  constraint_fn=constraint_func,
                                  swap_mutationRate=0.03,
                                  size_mutationRate=0.35, velocity=velocity,
                                  generations=500)
