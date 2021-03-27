###########################################################################
# In main directory
# Usage: '$ python code/branch_and_bound/satisfaction_opti.py'
#
# Check '$ python code/branch_and_bound/satisfaction_opti.py -h' for help
###########################################################################
# Imports
###########################################################################
# Standard library imports
import __init__
import os
import copy
import time
import argparse
import numpy as np
from datetime import datetime
from queue import PriorityQueue

# Local import
from code.data_input.base_input import TestCaseLoader

###########################################################################
# Code
###########################################################################
global OUTPUT_DIR
OUTPUT_DIR = os.path.join(
    os.getcwd(), 'output', 'branch_and_bound'
)
if os.path.exists(OUTPUT_DIR) is False:
    os.mkdir(OUTPUT_DIR)

INF = np.infty
N = 5  # Number of exhibits


def cmd_line_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ext', dest='ext', type=str, default='',
        help='Add a prefix to the plots, summary_data and summary_log '
        'before saving it'
    )

    parser.add_argument(
        '--tcn', dest='tc_number', type=int,
        default=0, help='Test case number'
    )

    parser.add_argument(
        '-d', dest='output_dir', type=str,
        default='BnB', help='Output folder name'
    )

    parser.add_argument(
        '-S', '--stamina', dest='stamina', type=float,
        default='1', help='Relation between travel penalty and popularity'
    )

    args = parser.parse_args()
    return args


def make_output_dir(folder_name, OUTPUT_DIR=OUTPUT_DIR):
    output_dir = os.path.join(OUTPUT_DIR, folder_name)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    return output_dir


def function_call_counter(func):
    '''
        Count number of times a function was called. To be used as decorator.
    '''
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


def function_timer(func):
    """
        Calculates the time required for function execution.
    """
    def wrapper(*args, **kwargs):
        begin = time.time()
        f = func(*args, **kwargs)
        end = time.time()
        wrapper.time_taken = end - begin
        print("\nTotal time taken in : ", func.__name__, wrapper.time_taken)
        return f

    wrapper.time_taken = 0
    return wrapper


class Node():
    """State Space Tree nodes
    """

    def __init__(self, tour, reduced_matrix, cost, Id, level):
        # stores edges of the state-space tree; help in tracing the path when
        # the answer is found
        self.tour = copy.deepcopy(tour)
        self.reduced_matrix = copy.deepcopy(reduced_matrix)
        self.cost = cost  # stores the lower bound
        self.Id = Id  # vertex -> stores the current city number
        self.level = level  # stores the total number of cities visited so far

    def __gt__(self, other):
        if(self.cost > other.cost):
            return True
        else:
            return False

    def __lt__(self, other):
        if(self.cost < other.cost):
            return True
        else:
            return False

    def __eq__(self, other):
        if(self.cost == other.cost):
            return True
        else:
            return False

    def __ne__(self, other):
        if(self.cost != other.cost):
            return True
        else:
            return False

    def debug(self):
        print("Tour: {} | Cost = {} | Node = {} | Level = {} |".format(
            self.tour, self.cost, self.Id, self.level), end='\r')


def CreateNode(parent_matrix, tour, level, i, j):
    """
        Function to allocate a new node `(i, j)` corresponds to visiting city
        `j` from city `i`

        Args:
            parent_matrix (N*N matrix): penalty matrix
            tour (list of [i,j]): edges visited till the node
            level (int): the total number of cities visited so far
            i (int): come from node Id
            j (int): goto node Id

        Returns:
            Node
    """
    node = Node(tour, [], 0, 0, 0)
    if level != 0:  # skip for the root node
        node.tour.append([i, j])
    node.reduced_matrix = copy.deepcopy(parent_matrix)

    # Change all entries of row `i` and column `j` to `INFINITY`
    # skip for the root node
    if level != 0:
        for k in range(N):
            node.reduced_matrix[i][k] = INF
            node.reduced_matrix[k][j] = INF

    # Set `(j, 0)` to `INFINITY`
    # here start node is 0
    node.reduced_matrix[j][0] = INF

    # set number of cities visited so far
    node.level = level

    # assign current city number
    node.Id = j

    return node


@function_call_counter
def matrix_reduction(node):
    # reduce each row so that there must be at least one zero in each row
    # node.reduced_matrix
    row = INF * np.ones(N)

    # `row[i]` contains minimum in row `i`
    for i in range(N):
        for j in range(N):
            if node.reduced_matrix[i][j] < row[i]:
                row[i] = node.reduced_matrix[i][j]

    # reduce the minimum value from each element in each row
    for i in range(N):
        for j in range(N):
            if node.reduced_matrix[i][j] != INF and row[i] != INF:
                node.reduced_matrix[i][j] -= row[i]

    # reduce each column so that there must be at least one zero in each column
    # node.reduced_matrix
    col = INF * np.ones(N)

    # `col[j]` contains minimum in col `j`
    for i in range(N):
        for j in range(N):
            if node.reduced_matrix[i][j] < col[j]:
                col[j] = node.reduced_matrix[i][j]

    # reduce the minimum value from each element in each column
    for i in range(N):
        for j in range(N):
            if node.reduced_matrix[i][j] != INF and col[j] != INF:
                node.reduced_matrix[i][j] -= col[j]

    # get the lower bound on the path starting at the current minimum node
    cost = 0

    for i in range(N):
        if row[i] != INF:
            cost += row[i]
        if col[i] != INF:
            cost += col[i]

    node.cost = cost


@function_timer
def solve(cost_matrix):
    # Create a priority queue to store live nodes of the search tree
    live_nodes = PriorityQueue()

    tour = []

    # The TSP starts from the first city, i.e., node 0
    root = CreateNode(cost_matrix, tour, 0, -1, 0)

    # get the lower bound of the path starting at node 0
    matrix_reduction(root)

    live_nodes.put((root.cost, root))  # add root to the list of live nodes

    while not live_nodes.empty():
        # a live node with the least estimated cost is selected
        minimum = live_nodes.get()[1]
        # minimum.debug()

        i = minimum.Id  # `i` stores the current city number

        # if all cities are visited; termination of loop
        if minimum.level == N - 1:
            minimum.tour.append([i, 0])  # return to starting city
            return minimum  # final node
            break

        # do for each child of min
        # `(i, j)` forms an edge in a space tree
        for j in range(N):
            if minimum.reduced_matrix[i][j] != INF:
                # create a child node and calculate its cost
                branch_node = CreateNode(
                    minimum.reduced_matrix, minimum.tour,
                    minimum.level + 1, i, j
                )

                # calculate the cost
                matrix_reduction(branch_node)

                branch_node.cost += minimum.cost + minimum.reduced_matrix[i][j]

                # # For debugging
                # print(
                # "Branch node cost: ", branch_node.cost, "minimum.cost",
                # minimum.cost, "minimum.reduced_matrix[i][j]",
                # minimum.reduced_matrix[i][j]
                # )

                # added the child to list of live nodes
                live_nodes.put((branch_node.cost, branch_node))

        del minimum


def print_tour(node):
    if node.level == N - 1:
        print("\nThe optimal tour is:")
        for i in range(N):
            print(node.tour[i][0], "-->", node.tour[i][1])
    else:
        print(node.tour)


def print_summary(output_dir, node, stamina, tc_name=None, ext=''):
    '''
        Prints the solver summary, and stores it in a `.log` file.

        Parameters:
        -----------
        save = (Boolean), default=True
            If True, saves the metadata in a log file
    '''

    try:
        logname = os.path.join(output_dir, 'BnB_satisfaction_summary.log')
        if os.path.exists(logname):
            os.remove(logname)

        outputFile = open(logname, 'a')

        def printing(text):
            print(text)
            if outputFile:
                outputFile.write(f'{text}\n')

        time_taken, reduction_calls = solve.time_taken, matrix_reduction.calls
        printing('\n===================================================')
        printing('Solver Summary:')
        printing('===================================================')
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pc_name = os.environ['COMPUTERNAME']
        printing(f'\nRun on: {dt_string} | PC: {pc_name}')
        if tc_name is not None:
            printing(f'Test case name: {tc_name}')

        printing(f'\nSolved in: {time_taken} s')
        printing(f'Number of reduction function calls: {reduction_calls}')

        print("\nThe optimal tour is:")
        for i in range(N):
            print(node.tour[i][0], "-->", node.tour[i][1])

        print("\nThe penalty incurred is {} for stamina {}".format(
            node.cost, stamina))

        printing('\n===================================================\n')

        print(f'Log file saved at: {logname}')

        if True:
            fname = os.path.join(
                output_dir, f'{ext}BnB_satisfaction_results.npz')
            np.savez(
                fname, time_taken=time_taken, func_calls=reduction_calls,
                opt_cost=node.cost
            )
            print(f'\nSummary data saved at: {fname}')
    finally:
        outputFile.close()


def main():
    # Read data off of standard library
    loader = TestCaseLoader()

    # Parse command line arguments
    args = cmd_line_parser()
    ext = args.ext
    stamina = args.stamina

    # Make output directory
    output_dir = make_output_dir(args.output_dir)

    tc_number = args.tc_number
    tc_name, cost_matrix = loader.get_test_data(tc_number)

    # COST_MATRIX = cost_matrix

    tc_name = "Manual input"

    COST_MATRIX = [
        [INF, 10, 8, 9, 7],
        [10, INF, 10, 5, 6],
        [8, 10, INF, 8, 9],
        [9, 5, 8, INF, 6],
        [7, 6, 9, 6, INF]
    ]  # optimal cost is 34

    # COST_MATRIX = [
    #     [INF, 3, 1, 5, 8],
    #     [3, INF, 6, 7, 9],
    #     [1, 6, INF, 4, 2],
    #     [5, 7, 4, INF, 3],
    #     [8, 9, 2, 3, INF]
    # ]  # optimal cost is 16

    # COST_MATRIX = [
    #     [INF, 1, 1, 1, 1],
    #     [1, INF, 1, 1, 1],
    #     [1, 1, INF, 1, 1],
    #     [1, 1, 1, INF, 1],
    #     [1, 1, 1, 1, INF]
    # ]

    # COST_MATRIX = [
    #     [INF, 199, 199, 19, 199],
    #     [199, INF, 199, 19, 199],
    #     [199, 199, INF, 19, 199],
    #     [199, 199, 199, INF, 199],
    #     [199, 199, 199, 19, INF]
    # ]

    # COST_MATRIX = [
    #     [INF, 2, 1, INF],
    #     [2, INF, 4, 3],
    #     [1, 4, INF, 2],
    #     [INF, 3, 2, INF]
    # ]  # optimal cost is 8

    # POPULARITY_INDEX = np.array([2,3,4,4])  # for N = 4
    POPULARITY_INDEX = np.array([2, 20, 2, 2, 2])  # for N = 5

    # `N` is the total number of total nodes on the graph or cities on the map
    global N
    COST_MATRIX = np.array(COST_MATRIX)
    N = len(COST_MATRIX)

    # Person cannot travel from one node to the same node
    for i in range(N):
        COST_MATRIX[i][i] = INF

    # Person cannot travel on restricted edges
    for i in range(N):
        for j in range(N):
            if COST_MATRIX[i][j] == 0:
                COST_MATRIX[i][j] = INF

    # Relation between travel penalty and popularity

    # higher the stamina, lower is the weightage of travel penalty
    STAMINA = stamina

    for i in range(N):
        for j in range(N):
            COST_MATRIX[i][j] -= STAMINA * POPULARITY_INDEX[j]

    print(COST_MATRIX)

    final_node = solve(COST_MATRIX)
    optimal_cost = final_node.cost

    # print_tour(final_node)
    # print("\nThe penalty incurred is {} for stamina {}".format(
    # optimal_cost, stamina))

    if True:
        print_summary(output_dir, final_node, stamina,
                      tc_name=tc_name, ext=ext)


if __name__ == '__main__':
    main()
