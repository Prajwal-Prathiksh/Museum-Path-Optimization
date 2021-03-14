###########################################################################
# Imports
###########################################################################
# Standard library imports
from pickle import TRUE
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from numpy.core.numeric import True_
sys.path.insert(0, os.getcwd())  # Insert this when you have any local imports

###########################################################################
# Code
###########################################################################
OUTPUT_DIR = os.path.join(
    os.getcwd(), 'output', 'simulated_annealing'
)

def function_calls(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

@function_calls
def get_cost_between_two_points(p1, p2, cost_matrix):
    '''
        Calculates the cost between two nodes from the cost matrix.

        Parameters:
        -----------
        p1: (int)
            Node 1 - index
        p2: (int)
            Node 2 - index
        cost_matrix: (Square Array)

        Returns:
        --------
        cost: (float)
            Cost between two nodes
    '''
    # Calculate cost
    cost = cost_matrix[p1][p2]
    return cost

@function_calls
def get_total_cost(node_list, cost_matrix):
    '''
        Calculates the cost of the entire node list.

        Parameters:
        -----------
        node_list: (List)
            List of nodes which are to be traversed
        cost_matrix: (Square Array)

        Returns:
        --------
        total_cost: (float)
            Total cost of the entire node list
    '''
    cost = 0.0
    for first, second in zip(node_list[:-1], node_list[1:]):
        # Distance between successive exhibits are added
        cost += get_cost_between_two_points(first, second, cost_matrix)

    # Distance between the first and the last exhibits is added
    cost += get_cost_between_two_points(
        node_list[0], node_list[-1], cost_matrix
    )
    return cost


class Coordinate:
    '''
        Coordinate Class
        ----------------
        A simple coordinate class, which represents the location of the exhibit
        which should be visited by the user.

        Parameters:
        -----------
        x: (float)
            x-coordinate of the exhibit
        y: (float)
            y-coordinate of the exhibit
    '''

    def __init__(self, x, y):
        '''
            Parameters:
            -----------
            x: (float)
                x-coordinate of the exhibit
            y: (float)
                y-coordinate of the exhibit
        '''
        self.x = x
        self.y = y

    @staticmethod
    def get_distance(p1, p2):
        '''
            Calculates the distance between two instances of the Coordinate
            class.

            Parameters:
            -----------
            p1: (Coordinate - Class)
                Exhibit 1
            p2: (Coordinate - Class)
                Exhibit 2

            Returns:
            --------
            dist: (float)
                Distance between two coordinates
        '''
        # Calculate distance
        dist = np.sqrt(np.abs(p1.x - p2.x) + np.abs(p1.y - p2.y))
        return dist

    @staticmethod
    def get_total_distance(coords):
        '''
            Calculates the total path distance in a list of instances of the
            Coordinate class. The sequence of the list represents the order in
            which the exhibits are to be covered by the tourist.

            Parameters:
            -----------
            coords: (List)
                List of Coordinate classes

            Returns:
            --------
            dist: (float)
                Total travel distance
        '''
        dist = 0.0
        for first, second in zip(coords[:-1], coords[1:]):
            # Distance between successive exhibits are added
            dist += Coordinate.get_distance(first, second)

        # Distance between the first and the last exhibits is added
        dist += Coordinate.get_distance(coords[0], coords[-1])
        return dist

    @staticmethod
    def random_coordinates_list(n, seed=None):
        '''
            Generate a list of instances of Coordinate class with random
            coordinates.

            Parameters:
            -----------
            n: (int)
                Length of output list
            seed: (int), default=None
                Reseed random number generator

            Returns:
            --------
            coords: (List)
                List of Coordinate classes
        '''
        if seed is None:
            np.random.seed()
        else:
            np.random.seed(0)

        coords = []
        for i in range(n):
            coords.append(Coordinate(np.random.uniform(), np.random.uniform()))

        return coords

    @staticmethod
    def plot_solution(initial_coords, optim_coords, save=True):
        '''
            Plots the inital and the optimized solution in a convinient format.

            Parameters:
            -----------
            initial_coords: (List)
                Inital list of Coordinate classes
            optim_coords: (List)
                Optimized list of Coordinate classes
            save = (Boolean), default=True
                If True, saves the plot

            Returns:
            --------
            Plot
        '''
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Initial Solution
        for first, second in zip(initial_coords[:-1], initial_coords[1:]):
            ax1.plot([first.x, second.x], [first.y, second.y], 'b')
        ax1.plot([initial_coords[0].x, initial_coords[-1].x],
                 [initial_coords[0].y, initial_coords[-1].y], 'b')

        for c in initial_coords:
            ax1.plot(c.x, c.y, 'ro')

        # Optimized Solution
        for first, second in zip(optim_coords[:-1], optim_coords[1:]):
            ax2.plot([first.x, second.x], [first.y, second.y], 'b')
        ax2.plot([optim_coords[0].x, optim_coords[-1].x],
                 [optim_coords[0].y, optim_coords[-1].y], 'b')

        for c in optim_coords:
            ax2.plot(c.x, c.y, 'ro')

        old_cost = round(Coordinate.get_total_distance(initial_coords), 2)
        new_cost = round(Coordinate.get_total_distance(optim_coords), 2)

        ax1.title.set_text(f'Initial Solution | Cost = {old_cost}')
        ax2.title.set_text(f'Optimized Solution | Cost = {new_cost}')

        if save == True:
            fname = os.path.join(OUTPUT_DIR, 'SSA_optimized_solution.png')
            plt.savefig(fname, dpi=400, bbox_inches='tight')

        plt.show()


class SimpleSimulatedAnnealing:
    '''
        Simple Simulated Annealing Class
        ---------------------------------
        A simple version of the Simulated Annealing algorithm.

        Parameters:
        -----------
        func0: (Function)
            Cost function to be evaluated
        x0: (Array):
            Initial solution
        T0: (float)
            Inital temperature
        alpha: (float)
            Cooling factor
        epochs: (int)
            Number of epochs
        N: (int)
            Number of iterations per epoch
        **kwargs:
            Additional arguments for `func0`
    '''

    def __init__(self, func0, x0, T0, alpha, epochs, N_per_epochs, **kwargs):
        '''
            Parameters:
            -----------
            func0: (Function)
                Cost function to be evaluated
            x0: (Array):
                Initial solution
            T0: (float)
                Inital temperature
            alpha: (float)
                Cooling factor
            epochs: (int)
                Number of epochs
            N_per_epochs: (int)
                Number of iterations per epoch
            **kwargs:
                Additional arguments for `func0`
        '''
        self.func0 = func0
        self.cost0 = round(self.func0(x0, **kwargs), 3)
        self.func0_calls = 0 # Set number of function calls to zero
        self.x0 = x0.copy()
        self.T0 = T0
        self.alpha = alpha
        self.epochs = epochs
        self.N_per_epochs = N_per_epochs

        self.len_x0 = len(x0)

        # Run Algorithm
        self.xf, self.cost_hist, self.rt = self.run_algorithm(**kwargs)
        self.costf = round(self.cost_hist[-1], 3)

        self.reduction_in_cost = round(np.abs(1 - self.cost0/self.costf)*100, 3)
        
        self.func0_calls = self.func0.calls

    def cooling_func(self, T):
        '''
            A simple cooling function.

            Parameters:
            -----------
            T: (float)
                Current temperature

            Returns:
            --------
            T_cooled: (float)
                Cooled temperature
        '''
        T_cooled = T * self.alpha
        return T_cooled

    def plot_cost_hist(self, ext='', save=True):
        '''
            Plot the history of cost of the objective function per epoch.

            Parameters:
            -----------
            ext: (string), default=''
                Add a prefix to the figure name before saving it
            save = (Boolean), default=True
                If True, saves the plot

            Returns:
            --------
            Plot
        '''
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)

        x = np.arange(1, self.epochs + 1)
        ax1.plot(x, self.cost_hist)

        ax1.title.set_text(f'Cost vs Epoch | Runtime: {self.rt} s')
        ax1.set(xlabel=r'Epochs $\rightarrow$', ylabel='Cost')

        if save == True:
            fname = os.path.join(OUTPUT_DIR, 'figures', f'{ext}SSA_cost_hist.png')
            plt.savefig(fname, dpi=400, bbox_inches='tight')

        plt.show()

    def run_algorithm(self, **kwargs):
        '''
            A simple Simulated Annealing algorithm.

            Returns:
            --------
            x: (Array)
                Optimized solution
            cost_host: (Array)
                History of cost of the objective function
            rt: (float)
                Runtime of the algorithm in fractional seconds
        '''
        x = self.x0.copy()
        cost0 = self.func0(x, **kwargs)
        T = self.T0

        np.random.seed()  # Reset seed

        cost_hist = []
        tic = time.monotonic()
        for epoch in range(self.epochs):
            # Store history of cost
            cost_hist.append(cost0)

            print(f'Epoch: {epoch} | Cost = {round(cost0, 3)}')

            T = self.cooling_func(T)

            for i in range(self.N_per_epochs):
                # Exchange two elements and get a new neighbour solution
                e1, e2 = np.random.randint(0, self.len_x0, size=2)
                temp = x[e1]
                x[e1] = x[e2]
                x[e2] = temp

                # Get the new cost
                cost1 = self.func0(x, **kwargs)

                if cost1 < cost0:
                    cost0 = cost1
                else:
                    if np.random.uniform() < np.exp((cost0 - cost1) / T):
                        cost0 = cost1
                    else:
                        # Re-swap
                        temp = x[e1]
                        x[e1] = x[e2]
                        x[e2] = temp

        toc = time.monotonic()
        rt = toc - tic

        cost_hist = np.array(cost_hist)
        return x, cost_hist, round(rt, 3)

    @staticmethod
    def factorial_apprx(N):
        '''
            Find the approximate of the factorial of a large number!

            Parameters:
            -----------
            N: (int)
                Number for which the approximate factorial needs to be calculated

            Returns:
            --------
            res: (string)
                Approximate factorial        
        '''
        P = np.math.factorial(N)
        res = int(np.floor(np.math.log10(P)))
        P = str(P)
        res = f'{P[0]}.{P[1]}{P[2]}e{res}'
        
        return res

    @staticmethod
    def solver_metadata(optim_solution, ext='', save=True):
        '''
            Prints the solver metadata, and stores it in a `.npz` file.

            Parameters:
            -----------
            optim_solution: (Class - SimpleSimulatedAnnealing)
                Instance of the `SimpleSimulatedAnnealing` class
            ext: (string), default=''
                Add a prefix to the `.npz` file before saving it
            save = (Boolean), default=True
                If True, saves the metadata in a `.npz` file
        '''
        rt, func0_calls = optim_solution.rt, optim_solution.func0.calls
        print(f'\nSolved in: {rt} s')
        print(f'Number of cost function calls: {func0_calls}')

        x0_len = len(optim_solution.x0)
        print(f'\nTotal number of nodes: {x0_len}')
        permut = SimpleSimulatedAnnealing.factorial_apprx(x0_len)
        print(f'Total permutations: {permut}')

        x0, xf = optim_solution.x0, optim_solution.xf
        cost0, costf = optim_solution.cost0, optim_solution.costf
        print(
            f'\nInital Cost: {cost0} ---> Optimized Cost: {costf}'
        )
        reduction_in_cost = optim_solution.reduction_in_cost
        print(f'Redcution in cost (in %): {reduction_in_cost} %')

        T0, alpha = optim_solution.T0, optim_solution.alpha
        epochs, N_per_epochs = optim_solution.epochs, optim_solution.N_per_epochs

        if save == True:
            fname = os.path.join(OUTPUT_DIR, f'{ext}SSA_results.npz')
            np.savez(
                fname, rt=rt, func0_calls=func0_calls, x0_len=x0_len, 
                permut=permut, x0=x0, cost0=cost0, xf=xf, costf=costf, 
                reduction_in_cost=reduction_in_cost, T0=T0, alpha=alpha, 
                epochs=epochs, N_per_epochs=N_per_epochs
            )


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    from code.data_input.base_input import BaseInputLoader

    # Set-up parameters for the Simulated Annealing Algorithm
    T0, alpha, epochs, N_per_epochs = [40, 0.99, 1000, 300]

    # ---------------------------------------------------------------------
    # Generate random coordinates
    # initial_coords = Coordinate.random_coordinates_list(80, seed=0)

    # Set up Simulated Annealing Class
    # optim_solution = SimpleSimulatedAnnealing(
    #     func0=Coordinate.get_total_distance, x0=initial_coords, T0=T0,
    #     alpha=alpha, epochs=epochs, N=N_per_epochs
    # )

    # optim_solution.plot_cost_hist()
    # Coordinate.plot_solution(initial_coords, optim_solution.xf)

    # ---------------------------------------------------------------------

    # Read data off of standard library
    fpath = os.path.join(os.getcwd(), 'code', 'data_input', 'test_load_list')
    loader = BaseInputLoader(fpath)

    fpath = os.path.join(os.getcwd(), 'code', 'data_input', 'test_load_list')
    loader = BaseInputLoader(fpath)
    cost_matrix = loader.get_input_test_case(1).get_cost_matrix()

    num_nodes = np.shape(cost_matrix)[0]

    np.random.seed(0)
    initial_soln = np.random.permutation(np.arange(num_nodes))

    # Set up Simulated Annealing Class
    optim_solution = SimpleSimulatedAnnealing(
        func0=get_total_cost, x0=initial_soln, T0=T0,
        alpha=alpha, epochs=epochs, N_per_epochs=N_per_epochs, cost_matrix=cost_matrix
    )

    SAVE = True
    SimpleSimulatedAnnealing.solver_metadata(optim_solution, save=SAVE)
    optim_solution.plot_cost_hist(ext='node_', save=SAVE)
