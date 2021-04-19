###########################################################################
# In main directory
# Usage: '$ python code/branch_and_bound/post_processing.py'
###########################################################################
# Imports
###########################################################################
import __init__
import numpy as np
import random

##############################################################################
# Code
##############################################################################


def get_input(num_of_nodes, fetch_coordinates=True, seed=0):
    random.seed(seed)
    coordi = np.round(np.random.rand(num_of_nodes, 2) * 10, 1)
    cost_matrix = np.zeros((num_of_nodes, num_of_nodes))

    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            cost_matrix[i, j] = np.sqrt(
                (coordi[i, 0] - coordi[j, 0]) ** 2 +
                (coordi[i, 1] - coordi[j, 1]) ** 2
            )

    if fetch_coordinates:
        return cost_matrix, coordi
    else:
        return cost_matrix


def main():
    cost_matrix, coordi = get_input(4)
    print("Cost matrix:", cost_matrix)
    print("List of co-ordinates:", coordi)


if __name__ == "__main__":
    main()
