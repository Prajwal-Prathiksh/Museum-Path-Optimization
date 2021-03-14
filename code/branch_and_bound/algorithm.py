import copy
import numpy as np
from queue import PriorityQueue

INF = np.infty
N = 5  # Number of exhibits


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

    def debug(self):
        print("Tour: {} has a cost {} for node {} at level {}".format(
            self.tour, self.cost, self.Id, self.level))


def CreateNode(parent_matrix, tour, level, i, j):
    """Function to allocate a new node `(i, j)` corresponds to visiting city `j` from city `i`

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


# to be editted
def print_tour(node):
    if node.level == N - 1:
        print()
        print("The optimal tour is:")
        for i in range(N):
            print(node.tour[i][0], "-->", node.tour[i][1])
    else:
        print(node.tour)


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
        minimum.debug()

        i = minimum.Id  # `i` stores the current city number

        # if all cities are visited; termination of loop
        if minimum.level == N - 1:
            minimum.tour.append([i, 0])  # return to starting city
            print_tour(minimum)
            return minimum.cost  # optimal cost
            break

        # do for each child of min
        # `(i, j)` forms an edge in a space tree
        for j in range(N):
            if minimum.reduced_matrix[i][j] != INF:
                # create a child node and calculate its cost
                branch_node = CreateNode(
                    minimum.reduced_matrix, minimum.tour, minimum.level + 1, i, j)

                # calculate the cost
                matrix_reduction(branch_node)

                branch_node.cost += minimum.cost + minimum.reduced_matrix[i][j]

                # # For debugging
                # print("Branch node cost: ", branch_node.cost, "minimum.cost", minimum.cost,
                #       "minimum.reduced_matrix[i][j]", minimum.reduced_matrix[i][j])

                # added the child to list of live nodes
                live_nodes.put((branch_node.cost, branch_node))

        del minimum


def main():

    # COST_MATRIX = [
    #     [INF, 10, 8, 9, 7],
    #     [10, INF, 10, 5, 6],
    #     [8, 10, INF, 8, 9],
    #     [9, 5, 8, INF, 6],
    #     [7, 6, 9, 6, INF]
    # ]  # optimal cost is 34

    # COST_MATRIX = [
    #     [INF, 3, 1, 5, 8],
    #     [3, INF, 6, 7, 9],
    #     [1, 6, INF, 4, 2],
    #     [5, 7, 4, INF, 3],
    #     [8, 9, 2, 3, INF]
    # ]  # optimal cost is 16

    COST_MATRIX = [
        [INF, 2, 1, INF],
        [2, INF, 4, 3],
        [1, 4, INF, 2],
        [INF, 3, 2, INF]
    ]  # optimal cost is 8

    # `N` is the total number of total nodes on the graph or cities on the map
    global N
    N = len(COST_MATRIX)

    print("Total cost is {}".format(solve(COST_MATRIX)))


if __name__ == '__main__':
    main()
