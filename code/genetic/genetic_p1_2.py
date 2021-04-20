import __init__
import numpy as np
import time
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from code.data_input.input_final import get_input_loader


class City:
    def __init__(self, x, y, index=0):
        self.x = x
        self.y = y
        self.index = index

    def distance(self, city):
        global COST_MATRIX
        return COST_MATRIX[self.index][city.index]

    def distance_old(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
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


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(
    population, popSize, eliteSize, mutationRate, generations
):
    tic = time.monotonic()

    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        temp = 1 / rankRoutes(pop)[0][1]

        print(f"Gen: {i} | Progress: {round(temp, 2)}", end='\r')
        progress.append(temp)

    print("\n\nFinal distance: " + str(1 / rankRoutes(pop)[0][1]))
    toc = time.monotonic()
    rt = toc - tic
    print(f'Runtime : {round(rt, 3)}')

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    return bestRoute


if __name__ == '__main__':

    tc_sym = 'asym'
    tc_number = 2

    if tc_sym == 'sym':
        tc_fname = 'Choose_TC_Sym_NPZ.txt'
    elif tc_sym == 'asym':
        tc_fname = 'Choose_TC_Asym_NPZ.txt'

    loader = get_input_loader(tc_fname, False)
    tc_name = loader.get_test_case_name(tc_number)
    cost_matrix = loader.get_input_test_case(tc_number).get_cost_matrix()

    cityNumber = np.shape(cost_matrix)[0]
    cityList = []
    for i in range(0, cityNumber):
        cityList.append(City(x=int(random.random() * 200),
                             y=int(random.random() * 200), index=i))
    global COST_MATRIX
    COST_MATRIX = [[0 for i in range(cityNumber)] for j in range(cityNumber)]
    for i in range(cityNumber - 1):
        COST_MATRIX[i][i] = np.inf
        for j in range(i + 1, cityNumber):
            x = int(1 + random.random() * 9)
            COST_MATRIX[i][j] = x
            COST_MATRIX[j][i] = x

    COST_MATRIX = cost_matrix  # data['cost_matrix']

geneticAlgorithm(
    population=cityList,
    popSize=150,
    eliteSize=15,
    mutationRate=0.02,
    generations=200
)
