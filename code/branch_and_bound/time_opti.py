###########################################################################
# In main directory
# Usage: '$ python code/branch_and_bound/time_opti.py'
#
# Check '$ python code/branch_and_bound/time_opti.py -h' for help
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
from numba import njit
from datetime import datetime
from queue import PriorityQueue

# Local imports
from code.data_input.input_final import get_input_loader
from code.branch_and_bound.get_input import get_input


###########################################################################
# Code
###########################################################################
global OUTPUT_DIR
OUTPUT_DIR = os.path.join(os.getcwd(), "output", "branch_and_bound")
if os.path.exists(OUTPUT_DIR) is False:
    os.mkdir(OUTPUT_DIR)

INF = np.infty
N = 5  # Number of exhibits


def cmd_line_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ext",
        dest="ext",
        type=str,
        default="",
        help="Add a prefix to the plots, summary_data and summary_log "
        "before saving it",
    )

    parser.add_argument(
        "--tcn", dest="tc_number", type=int, default=1, help="Test case number"
    )

    parser.add_argument(
        "-d",
        dest="output_dir",
        type=str,
        default="BnB",
        help="Output folder name",
    )

    args = parser.parse_args()
    return args


def make_output_dir(folder_name, OUTPUT_DIR=OUTPUT_DIR):
    output_dir = os.path.join(OUTPUT_DIR, folder_name)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    return output_dir


def function_call_counter(func):
    """
        Count number of times a function was called. To be used as decorator.
    """

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
        print("Total time taken in : ", func.__name__, wrapper.time_taken)
        return f

    wrapper.time_taken = 0
    return wrapper


class Node:
    """
    State Space Tree nodes (exhibits)
    """

    def __init__(self, tour, reduced_matrix, cost, Id, level):
        # stores edges of the state-space tree; help in tracing the path when
        # the answer is found
        self.tour = copy.deepcopy(tour)
        self.reduced_matrix = copy.deepcopy(reduced_matrix)
        self.cost = cost  # stores the lower bound
        self.Id = Id  # vertex -> stores the current node number
        self.level = level  # stores the total number of nodes visited so far

    def __gt__(self, other):
        if self.cost > other.cost:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.cost < other.cost:
            return True
        else:
            return False

    def debug(self, with_tour=False):
        if with_tour:
            print(
                "Level = {} | Cost = {} | Node = {} | Tour = {}".format(
                    self.level, self.cost, self.Id, self.tour
                )
            )
        else:
            print(
                "Level = {} | Cost = {} | Node = {}".format(
                    self.level, self.cost, self.Id
                )
            )


def CreateNode(parent_matrix, tour, level, i, j):
    """
        Function to allocate a new node `(i, j)` corresponds to visiting node
        `j` from node `i`

        Args:
            parent_matrix (N*N matrix): penalty matrix
            tour (list of [i,j]): edges visited till the node
            level (int): the total number of nodes visited so far
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

    # set number of nodes visited so far
    node.level = level

    # assign current node number
    node.Id = j

    return node


@function_call_counter
# @njit
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


@function_call_counter
@njit
def matrix_reduction_generic(reduced_matrix):
    # reduce each row so that there must be at least one zero in each row
    # node.reduced_matrix
    row = INF * np.ones(N)

    # `row[i]` contains minimum in row `i`
    for i in range(N):
        for j in range(N):
            if reduced_matrix[i][j] < row[i]:
                row[i] = reduced_matrix[i][j]

    # reduce the minimum value from each element in each row
    for i in range(N):
        for j in range(N):
            if reduced_matrix[i][j] != INF and row[i] != INF:
                reduced_matrix[i][j] -= row[i]

    # reduce each column so that there must be at least one zero in each column
    # node.reduced_matrix
    col = INF * np.ones(N)

    # `col[j]` contains minimum in col `j`
    for i in range(N):
        for j in range(N):
            if reduced_matrix[i][j] < col[j]:
                col[j] = reduced_matrix[i][j]

    # reduce the minimum value from each element in each column
    for i in range(N):
        for j in range(N):
            if reduced_matrix[i][j] != INF and col[j] != INF:
                reduced_matrix[i][j] -= col[j]

    # get the lower bound on the path starting at the current minimum node
    cost = 0

    for i in range(N):
        if row[i] != INF:
            cost += row[i]
        if col[i] != INF:
            cost += col[i]

    return cost, reduced_matrix


@function_timer
def solve(cost_matrix, is_tour_stored=False):
    # Create a priority queue to store live nodes of the search tree
    live_nodes = PriorityQueue()

    tour = []
    full_tour = []

    # The TSP starts from the first node, i.e., node 0
    root = CreateNode(cost_matrix, tour, 0, -1, 0)

    # get the lower bound of the path starting at node 0
    # matrix_reduction(root)
    root.cost, root.reduced_matrix = matrix_reduction_generic(
        root.reduced_matrix
    )

    live_nodes.put((root.cost, root))  # add root to the list of live nodes

    while not live_nodes.empty():
        # a live node with the least estimated cost is selected
        minimum = live_nodes.get()[1]
        # minimum.debug(with_tour=True)  # for debugging purposes

        if is_tour_stored:
            full_tour.append(copy.deepcopy(minimum.tour))

        i = minimum.Id  # `i` stores the current node number

        # if all nodes are visited; termination of loop
        if minimum.level == N - 1:
            minimum.tour.append([i, 0])  # return to starting node
            full_tour.append(minimum.tour)
            # print("Returning minimum node of type", type(minimum))
            return minimum, full_tour  # final node

        # do for each child of min
        # `(i, j)` forms an edge in a space tree
        for j in range(N):
            if minimum.reduced_matrix[i][j] != INF:
                # create a child node and calculate its cost
                branch_node = CreateNode(
                    minimum.reduced_matrix,
                    minimum.tour,
                    minimum.level + 1,
                    i,
                    j,
                )

                # calculate the cost
                # matrix_reduction(branch_node)
                (
                    branch_node.cost,
                    branch_node.reduced_matrix,
                ) = matrix_reduction_generic(branch_node.reduced_matrix)

                branch_node.cost += minimum.cost + minimum.reduced_matrix[i][j]

                # # For debugging
                # print(
                #     "Branch node cost: ",
                #     branch_node.cost
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


def print_summary(
    output_dir, node, full_tour=[], tc_name=None, ext="", coordi=None
):
    """
        Prints the solver summary, and stores it in a `.log` file.

        Parameters:
        -----------
        save = (Boolean), default=True
            If True, saves the metadata in a log file
    """

    try:
        logname = os.path.join(output_dir, "BnB_summary.log")
        if os.path.exists(logname):
            os.remove(logname)

        outputFile = open(logname, "a")

        def printing(text):
            print(text)
            if outputFile:
                outputFile.write(f"{text}\n")

        time_taken, reduction_calls = (
            solve.time_taken,
            matrix_reduction_generic.calls,
        )
        printing("\n===================================================")
        printing("Solver Summary:")
        printing("===================================================")
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pc_name = os.environ["COMPUTERNAME"]
        printing(f"\nRun on: {dt_string} | PC: {pc_name}")
        if tc_name is not None:
            printing(f"Test case name: {tc_name}")

        printing(f"\nSolved in: {time_taken} s")
        printing(f"Number of reduction function calls: {reduction_calls}")

        print("\nThe optimal tour is:")
        for i in range(N):
            print(node.tour[i][0], "-->", node.tour[i][1])

        print("\nTotal cost is {}".format(node.cost))

        printing("\n===================================================\n")

        print(f"Log file saved at: {logname}")

        if True:
            fname = os.path.join(output_dir, f"{ext}BnB_results.npz")
            np.savez(
                fname,
                time_taken=time_taken,
                func_calls=reduction_calls,
                opt_cost=node.cost,
                tour=node.tour,
                full_tour=np.array(full_tour, dtype="object"),
                coordi=coordi,
            )
            print(f"\nSummary data saved at: {fname}")
    finally:
        outputFile.close()


def main():
    # Read data off of standard library for symmetric
    loader = get_input_loader("Choose_TC_Sym_NPZ.txt", False)
    print("Solving symmetric problem...")

    # # Read data off of standard library for asymmetric
    # loader = get_input_loader("Choose_TC_Asym_NPZ.txt", False)
    # print("\nSolving asymmetric problem...")

    # Parse command line arguments
    args = cmd_line_parser()
    ext = args.ext

    # Make output directory
    output_dir = make_output_dir(args.output_dir)

    tc_number = args.tc_number
    tc_name = loader.get_test_case_name(tc_number)
    coordi = None
    cost_matrix = loader.get_input_test_case(tc_number).get_cost_matrix()

    COST_MATRIX = cost_matrix

    # tc_name = "Manual input"

    print("Test case name: ", tc_name)

    inf = INF
    # COST_MATRIX = np.load("data/manual/2floorcostmatrix.npy")

    # `N` is the total number of total nodes on the graph
    global N

    # N = 14
    # COST_MATRIX, coordi = get_input(N, fetch_coordinates=True)

    # COST_MATRIX = [
    #     [INF, 10, 8, 9, 7],
    #     [10, INF, 10, 5, 6],
    #     [8, 10, INF, 8, 9],
    #     [9, 5, 8, INF, 6],
    #     [7, 6, 9, 6, INF],
    # ]  # optimal cost is 34

    # COST_MATRIX = [
    #     [INF, 3, 1, 5, 8],
    #     [3, INF, 6, 7, 9],
    #     [1, 6, INF, 4, 2],
    #     [5, 7, 4, INF, 3],
    #     [8, 9, 2, 3, INF]
    # ]  # optimal cost is 16

    # COST_MATRIX = [
    #     [INF, 2, 1, INF],
    #     [2, INF, 4, 3],
    #     [1, 4, INF, 2],
    #     [INF, 3, 2, INF]
    # ]  # optimal cost is 8

    COST_MATRIX = np.array(COST_MATRIX)
    N = len(COST_MATRIX)

    # print(COST_MATRIX)

    # Person cannot travel from one node to the same node
    for i in range(N):
        COST_MATRIX[i][i] = INF

    # Person cannot travel on restricted edges
    for i in range(N):
        for j in range(N):
            if COST_MATRIX[i][j] == 0:
                COST_MATRIX[i][j] = INF

    print("Cost Matrix is:\n", COST_MATRIX)

    print("Number of nodes are {}".format(N))

    final_node, full_tour = solve(COST_MATRIX, is_tour_stored=True)
    # print(type(final_node))
    optimal_cost = final_node.cost

    # print_tour(final_node)
    if True:
        print_summary(
            output_dir,
            final_node,
            full_tour=full_tour,
            tc_name=tc_name,
            ext=ext,
            coordi=coordi,
        )

    print("Total cost is {}".format(optimal_cost))


if __name__ == "__main__":
    main()
