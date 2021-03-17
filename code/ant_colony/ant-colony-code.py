import numpy as np
from numpy import inf

# given values for the problems

Cij = np.array(
    [[0, 10, 12, 11, 14],
     [10, 0, 13, 15, 8],
     [12, 13, 0, 9, 14],
     [11, 15, 9, 0, 16],
     [14, 8, 14, 16, 0]]
)


iteration = 20
n_ants = 5
n_citys = 5
Tmax = 1000000  # given value of maximum time
velocity = 1  # for calculating time
t = 2  # constant given in Problem in time taken formula

# intialization part

m = n_ants
n = n_citys
Tm = Tmax
v = velocity
e = .5  # evaporation rate
alpha = 1  # pheromone factor
beta = 2  # visibility factor

# calculating the visibility of the next city visibility(i,j)=1/d(i,j)

visibility = 1 / Cij
visibility[visibility == inf] = 0

# initialize number of iterations

i = 0

# intializing pheromne present at the paths to the cities

pheromne = .1 * np.ones((n, n))

# intializing the route of the ants with size route(n_ants,n_citys+1)
# note adding 1 because we want to come back to the source city

visited = np.ones((m, n + 1))

for i in range(iteration):
    print('iteration =', i)

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

                combine_feature = np.multiply(
                    p_feature, v_feature)  # calculating the combine feature

                total = np.sum(combine_feature)  # sum of all the feature

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

    print(visited)
    route_opt = np.array(visited)  # intializing optimal route

    cost = np.zeros((m, 1))  # intializing total_distance_of_tour with zero

    for k in range(m):

        c = 0
        for j in range(n - 1):

            # calcualting total tour distance
            c = c + Cij[int(route_opt[k, j]) - 1, int(route_opt[k, j + 1]) - 1]

        cost[k] = c  # storing distance of tour for 'i'th ant at location 'i'

    dist_min_loc = np.argmin(cost)  # finding location of minimum of dist_cost
    dist_min_cost = cost[dist_min_loc]  # finging min of dist_cost

    # intializing current traversed as best route
    best_route = visited[dist_min_loc, :]
    pheromne = (1 - e) * pheromne  # evaporation of pheromne with (1-e)

    for k in range(m):
        for j in range(n - 1):
            dt = 1 / cost[k]
            pheromne[int(route_opt[k, j]) - 1, int(route_opt[k, j + 1]) - 1] =\
                pheromne[int(route_opt[k, j]) - 1,
                         int(route_opt[k, j + 1]) - 1] + dt
            # updating the pheromne with delta_distance
            # delta_distance will be more with min_dist i.e adding more weight
            # to that route  peromne

print('best path :', best_route)
print('cost of the best path', int(
    dist_min_cost[0]) + Cij[int(best_route[-2]) - 1, 0])
