#!/usr/bin/env python

#############
# Imports
#############
# Standard library imports
import __init__
from numpy import inf
import matplotlib.pyplot as plt
import numpy as np
import time

# Local import
from code.data_input.input_final import get_input_loader

#############
# Code
#############

# Write 'sym' for symmetric cases and 'asym' for asymmetric cases
tc_sym = 'sym'
tc_number = 1


if tc_sym == 'sym':
    tc_fname = 'Choose_TC_Sym_NPZ.txt'
elif tc_sym == 'asym':
    tc_fname = 'Choose_TC_Asym_NPZ.txt'

loader = get_input_loader(tc_fname, False)
tc_name = loader.get_test_case_name(tc_number)
cost_matrix = loader.get_input_test_case(tc_number).get_cost_matrix()

print('tc_name=', tc_name)
Cij = cost_matrix
# Cij = np.array([[0,10,12,11,14]
#           ,[10,0,13,15,8]
#           ,[12,13,0,9,14]
#           ,[11,15,9,0,16]
#           ,[14,8,14,16,0]])

iteration = 80  # no of iterations
n_ants = 5
n_citys = len(Cij[0])
Tmax = 10000000000  # given value of maximum time
velocity = 1  # for calculating time
t = 2  # constant given in Problem in time taken formula

# intialization part

m = n_ants
n = n_citys
Tm = Tmax
v = velocity
e = .3  # evaporation rate
alpha = 1  # pheromone factor
beta = 2  # visibility factor

# array that will be used in final graph for storing all cost values
cost_best_list = [0] * 4
# array that will be used in final graph for storing all iteration values
iteration_list = [0] * 4
# calculating the visibility of the next city visibility(i,j)=1/d(i,j)

visibility = 1 / Cij
visibility[visibility == inf] = 0

# initialize number of iterations

i = 0

# intializing pheromne present at the paths to the cities
count = 0
pheromne = .1 * np.ones((n, n))
for m in [10, 20, 30, 40]:
    pheromne = .1 * np.ones((n, n))
    visibility = 1 / Cij
    visibility[visibility == inf] = 0

    # array that will be used in final graph for storing all iteration values
    m_iteration_list = np.empty(0)
    m_cost_best_list = np.empty(0)
    time_start = time.time()
    # intializing the route of the ants with size route(n_ants,n_citys+1)
    # note adding 1 because we want to come back to the source city

    visited = np.ones((m, n + 1))

    for i in range(iteration):
        # print('iteration =', i)

        # initializing step counter
        s = 0

        visited[:, 0] = 1  # placing every ant on the entry

        # initialising total time taken marix
        T = np.zeros(m)

        # creating a copy of visibility
        temp_visibility = np.ones((m, n, n)) * visibility
        # print(temp_visibility)

        while s < n - 1:

            for k in range(0, m):

                if T[k] + Cij[int(visited[k, s - 1]) - 1, 0] / v + t <= Tm:

                    # intializing combine_feature array to zero
                    combine_feature = np.zeros(n)
                    # intializing cummulative probability array to zeros
                    cum_prob = np.zeros(n)

                    cur_loc = int(visited[k, s] - 1)  # current city of the ant

                    # making visibility of the current city as zero
                    temp_visibility[k][:, cur_loc] = 0
                    # calculating pheromne feature
                    p_feature = np.power(pheromne[cur_loc, :], beta)
                    # calculating visibility feature
                    v_feature = np.power(temp_visibility[k][cur_loc, :], alpha)
                    # print(p_feature)
                    # print(v_feature)
                    # adding axis to make a size[5,1]
                    p_feature = p_feature[:, np.newaxis]
                    # adding axis to make a size[5,1]
                    v_feature = v_feature[:, np.newaxis]

                    # calculating the combine feature
                    combine_feature = np.multiply(p_feature, v_feature)

                    total = np.sum(combine_feature)  # sum of all the feature
                    if total == 0:
                        probs = 1
                    else:
                        # finding probability of element probs(i) =
                        # comine_feature(i)/total
                        probs = combine_feature / total
                    # print(probs)
                    cum_prob = np.cumsum(probs)  # calculating cummulative sum
                    # print(cum_prob)
                    # print(cum_prob)
                    r = np.random.random_sample()  # random no in [0,1)
                    # print(r)
                    # print(r)
                    # finding the next city having probability higher then
                    # random(r)
                    exhibit = np.nonzero(cum_prob > r)[0][0] + 1
                    # print(city)

                    visited[k, s + 1] = exhibit  # adding city to route

                    T[k] = T[k] + Cij[int(visited[k, s]) - 1,
                                      int(visited[k, s + 1]) - 1] / v + t

                else:

                    visited[k, s + 1] = 1

                    if visited[k, s] != 0:
                        T[k] = T[k] + Cij[int(visited[k, s]) - 1,
                                          int(visited[k, s + 1]) - 1] / v + t

            s = s + 1

        # print(visited)
        route_opt = np.array(visited)  # intializing optimal route

        cost = np.zeros((m, 1))  # intializing total_distance_of_tour with zero

        for k in range(m):

            c = 0
            for j in range(n - 1):

                # calcualting total tour distance
                c = c + Cij[int(route_opt[k, j]) - 1,
                            int(route_opt[k, j + 1]) - 1]

            # storing distance of tour for 'i'th ant at location 'i'
            cost[k] = c

        # finding location of minimum of dist_cost
        dist_min_loc = np.argmin(cost)
        dist_min_cost = cost[dist_min_loc]  # finding min of dist_cost

        # intializing current traversed as best route
        best_route = visited[dist_min_loc, :]
        pheromne = (1 - e) * pheromne  # evaporation of pheromne with (1-e)
        cost_best_route = int(
            dist_min_cost[0]) + Cij[int(best_route[-2]) - 1, 0]
        m_cost_best_list = np.append(m_cost_best_list, cost_best_route)
        m_iteration_list = np.append(m_iteration_list, i + 1)
        # print(m_cost_best_list)
        for k in range(m):
            for j in range(n - 1):
                dt = 1 / cost[k]
                pheromne[int(route_opt[k, j]) - 1, int(route_opt[k, j + 1]) -
                         1] = pheromne[
                             int(route_opt[k, j]) - 1,
                             int(route_opt[k, j + 1]) - 1] + dt
                # updating the pheromne with delta_distance
                # delta_distance will be more with min_dist i.e adding more
                # weight to that route  peromne

    time_end = time.time()
    time_taken = time_end - time_start

    cost_best_list[count] = m_cost_best_list
    iteration_list[count] = m_iteration_list
    count += 1
    # print(iteration_list)
    print("Number of ants:", m)
    # print('best path :', best_route)
    print('cost of the best path', int(
        dist_min_cost[0]) + Cij[int(best_route[-2]) - 1, 0])
    print("Time taken to solve is {} sec".format(round(time_taken, 3)))
    print("Test case name: ", tc_name)

for i in range(4):
    plt.plot(iteration_list[i], cost_best_list[i])
plt.xlabel("number of iterations")
plt.ylabel("optimal cost")
plt.title("Optimal Cost vs Number of iterations (varying number of ants)")
plt.legend(["m=10", "m=20", "m=30", "m=40"])
plt.show()

# #Below is the code for plotting coordinates and creating colormap

# ##First we will create a random set of points, and plot them
# z = 12 #number of nodes wanted

# x = 100*np.random.rand(z)
# y = 100*np.random.rand(z)

# Cij = np.zeros([z,z])

# for i in range(z):
#   for j in range(z):
#     Cij[i,j] = ((x[i]-x[j])**2 + (y[i]-y[j])**2)**0.5

# #This Cij obtained should be used in the above code first

# After we have run our main code with this Cij, we can plot colormaps for
# Pheromone and Visbility

# plt.scatter(x, y)
# print(Cij)

# x1 = np.zeros(len(x)+1)
# y1 = np.zeros(len(x)+1)
# print(best_route)
# best_route = np.array([int(j) for j in best_route])
# for j in range(len(x)+1):
#   x1[j] = x[best_route[j]-1]
#   y1[j] = y[best_route[j]-1]
# #plt.plot(x1,y1)
# plt.quiver(
#     x1[:-1], y1[:-1], x1[1:] - x1[:-1], y1[1:] - y1[:-1], scale_units='xy',
#     angles='xy', scale=1
# )
# plt.annotate("Entry/Exit", (x1[0], y1[0]))
# plt.xlim(-5,95)
# plt.scatter(x,y)
# plt.title("Best Route")
# plt.show()

# #Above code plots the path

# import matplotlib.colors as col
# multiplied with 1000 because otherwise only one very light color was
# showing in colormap
# matrix = 1000*pheromne
# minima = np.array(matrix).min()
# maxima = np.array(matrix).max()

# norm = col.Normalize(vmin=minima, vmax=maxima, clip=True)

# colors = (plt.cm.YlOrRd(matrix))

# for d in range(z):
#   for e in range(z):
#     plt.plot([x[d],x[e]], [y[d],y[e]], color = colors[d,e])

# plt.scatter(x,y, c = 'b', s = 65)
# plt.title('Pheromone Levels')
# plt.xlim(-5,80)
# plt.show()

# #Above code prints pheromone levels

# visibility = 1/Cij
# visibility[visibility == inf] = 0

# multiplied with 20 because otherwise only one very light color was showing
# in colormap
# matrix = 20*visibility
# minima = np.array(matrix).min()
# maxima = np.array(matrix).max()

# norm = col.Normalize(vmin=minima, vmax=maxima, clip=True)

# colors = (plt.cm.YlGnBu(matrix))

# for d in range(z):
#   for e in range(z):
#     plt.plot([x[d],x[e]], [y[d],y[e]], color = colors[d,e])

# plt.scatter(x,y, c = 'r', s = 65)
# plt.title('Visibility')
# plt.xlim(-5,80)
# plt.show()

# #This code plots visibilty
