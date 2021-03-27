###########################################################################
# Imports
###########################################################################
# Standard library imports
import __init__
import argparse
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Local imports
from code.data_input.input_final import get_input_loader

###########################################################################
# Code
###########################################################################
global OUTPUT_DIR
OUTPUT_DIR = os.path.join(
    os.getcwd(), 'output', 'simulated_annealing'
)
if os.path.exists(OUTPUT_DIR) is False:
    os.mkdir(OUTPUT_DIR)

CFUNCS = ['simp', 'exp']


def make_output_dir(folder_name, OUTPUT_DIR=OUTPUT_DIR):
    if folder_name is None:
        output_dir = os.path.join(OUTPUT_DIR, 'SSA')
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        return output_dir

    else:
        if os.path.exists(folder_name) is False:
            os.mkdir(folder_name)
            OUTPUT_DIR = folder_name
        return folder_name


def cli_parser():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--T', action='store', dest='T', type=float,
        default=40, help='Inital temperature'
    )
    parser.add_argument(
        '--alpha', action='store', dest='alpha', type=float,
        default=0.99, help='Cooling factor'
    )
    parser.add_argument(
        '--epoch', action='store', dest='epochs', type=int,
        default=1000, help='Number of epochs'
    )
    parser.add_argument(
        '--n-epoch', action='store', dest='N_per_epochs', type=int,
        default=100, help='Number of iterations per epoch'
    )
    parser.add_argument(
        '--s', action='store_true', dest='SAVE',
        help='If true, stores any generated plots and the summary data'
    )
    parser.add_argument(
        '--cfunc', action='store', dest='cfunc', choices=CFUNCS,
        default='simp', help='Type of cooling function to be used'
    )
    parser.add_argument(
        '--ext', action='store', dest='ext', type=str,
        default='',
        help='Add a prefix to the plots, summary_data and summary_log '
        'before saving it'
    )
    parser.add_argument(
        '--tcn', action='store', dest='tc_number', type=int,
        default=1, help='Test case number'
    )
    parser.add_argument(
        '--sym', action='store', dest='tc_sym', choices=['sym', 'asym'],
        default='sym', help='Run symmetric or asymetric standard test cases'
    )
    parser.add_argument(
        '--d', action='store', dest='output_dir', type=str,
        default=None, help='Output folder name'
    )

    args = parser.parse_args()
    return args


def function_calls(f):
    '''
        Count number of times a function was called. To be used as decorator.
    '''
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
        # Distance between successive nodes are added
        cost += get_cost_between_two_points(first, second, cost_matrix)

    # Distance between the first and the last nodes is added
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
                List of Coordinate instances

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
                List of Coordinate instances
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
    def plot_solution(initial_coords, optim_coords, output_dir, save=True):
        '''
            Plots the inital and the optimized solution in a convinient format.

            Parameters:
            -----------
            initial_coords: (List)
                Inital List of Coordinate instances
            optim_coords: (List)
                Optimized List of Coordinate instances
            output_dir: (string)
                Absolute path of the output directory
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

        if save:
            fname = os.path.join(output_dir, 'SSA_optimized_solution.png')
            plt.savefig(fname, dpi=400, bbox_inches='tight')


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
        N_per_epochs: (int)
            Number of iterations per epoch
        output_dir: (string)
            Absolute path of the output directory
        cooling_func: (string), default=simple
            Type of cooling function to be used
            Choose from: ['simp', 'exp']
        ext: (string), default=''
            Add a prefix to the plots, summary_data and summary_log before
            saving it
        **kwargs:
            Additional arguments for `func0`
    '''

    def __init__(
        self, func0, x0, T0, alpha, epochs, N_per_epochs, output_dir,
        cooling_func='simp', ext='', **kwargs
    ):
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
            output_dir: (string)
                Absolute path of the output directory
            cooling_func: (string), default=simple
                Type of cooling function to be used
                Choose from: ['simp', 'exp']
            ext: (string), default=''
                Add a prefix to the plots, summary_data and summary_log before
                saving it
            **kwargs:
                Additional arguments for `func0`
        '''
        self.output_dir = output_dir
        # Initialize
        self.func0 = func0
        self.cost0 = round(self.func0(x0, **kwargs), 3)
        self.func0_calls = 0  # Set number of function calls to zero
        self.x0 = x0.copy()
        self.T0 = T0
        self.alpha = alpha
        self.epochs = epochs
        self.N_per_epochs = N_per_epochs
        self.ext = ext

        self.cooling_funcs_dict = dict(
            simp=self.simple_cooling_func, exp=self.exponential_cooling_func
        )

        self.cooling_func = self.cooling_funcs_dict[cooling_func]

        self.len_x0 = len(x0)

        # Print initial conditions
        print(f'\nTotal Number of Epochs: {self.epochs}')
        print(f'Initial Cost: {self.cost0}')
        # Run Algorithm
        self.xf, self.cost_hist, self.rt = self.run_algorithm(**kwargs)
        self.costf = round(self.cost_hist[-1], 3)

        self.reduction_in_cost = round(
            np.abs(1 - self.cost0 / self.costf) * 100, 3
        )

        self.func0_calls = self.func0.calls

    def simple_cooling_func(self, T, epoch_num):
        '''
            A simple cooling function.

            Parameters:
            -----------
            T: (float)
                Current temperature
            epoch_num: (int)
                Current epoch number

            Returns:
            --------
            T_cooled: (float)
                Cooled temperature
        '''
        T_cooled = T * self.alpha
        return T_cooled

    def exponential_cooling_func(self, T, epoch_num):
        '''
            A simple cooling function.

            Parameters:
            -----------
            T: (float)
                Current temperature
            epoch_num: (int)
                Current epoch number

            Returns:
            --------
            T_cooled: (float)
                Cooled temperature
        '''
        T_cooled = T * np.math.exp(-self.alpha * epoch_num)
        return T_cooled

    def plot_cost_hist(self, save=True):
        '''
            Plot the history of cost of the objective function per epoch.

            Parameters:
            -----------
            save = (Boolean), default=True
                If True, saves the plot

            Returns:
            --------
            Plot
        '''
        ext = self.ext
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)

        x = np.arange(1, self.epochs + 1)
        ax1.plot(x, self.cost_hist)

        ax1.title.set_text(f'Cost vs Epoch | Runtime: {self.rt} s')
        ax1.set(xlabel=r'Epochs $\rightarrow$', ylabel='Cost')

        if save:
            fname = os.path.join(
                self.output_dir, f'{ext}SSA_cost_hist.png'
            )
            plt.savefig(
                fname, dpi=400, bbox_inches='tight'
            )
            print(f'\nPlot saved at: {fname}')

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

            print(f'Epoch: {epoch} | Cost = {round(cost0, 3)}', end='\r')

            T = self.cooling_func(T, epoch)

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
                    if np.random.uniform() < np.math.exp((cost0 - cost1) / T):
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

    def solver_summary(self, tc_name=None, save=True):
        '''
            Prints the solver summary, and stores it in a `.npz` file.

            Parameters:
            -----------
            optim_solution: (Class - SimpleSimulatedAnnealing)
                Instance of the `SimpleSimulatedAnnealing` class
            save = (Boolean), default=True
                If True, saves the metadata in a `.npz` file
        '''
        ext = self.ext
        try:
            logname = os.path.join(self.output_dir, 'SSA_solver_summary.log')
            if os.path.exists(logname):
                os.remove(logname)

            outputFile = open(logname, 'a')

            def printing(text):
                print(text)
                if outputFile:
                    outputFile.write(f'{text}\n')

            rt, func0_calls = self.rt, self.func0.calls
            printing('\n===================================================')
            printing('Solver Summary:')
            printing('===================================================')
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            pc_name = os.environ['COMPUTERNAME']
            printing(f'\nRun on: {dt_string} | PC: {pc_name}')
            if tc_name is not None:
                printing(f'Test case name: {tc_name}')

            printing(f'\nSolved in: {rt} s')
            printing(f'Number of cost function calls: {func0_calls}')

            x0_len = len(self.x0)
            printing(f'\nTotal number of nodes: {x0_len}')
            permut = SimpleSimulatedAnnealing.factorial_apprx(x0_len)
            printing(f'Total permutations: {permut}')

            x0, xf = self.x0, self.xf
            cost0, costf = self.cost0, self.costf
            cost_hist = self.cost_hist
            printing(
                f'\nInitial Cost: {cost0} ---> Optimized Cost: {costf}'
            )
            reduction_in_cost = self.reduction_in_cost
            printing(f'Reduction in cost (in %): {reduction_in_cost} %')

            T0, alpha = self.T0, self.alpha
            epochs = self.epochs
            N_per_epochs = self.N_per_epochs
            cooling_func = self.cooling_func.__name__
            printing(f'\nT0 = {T0} | alpha = {alpha}')
            printing(f'epochs = {epochs} | iter/epoch = {N_per_epochs}')
            printing(f'Cooling Function: {cooling_func}()')

            printing('\n===================================================\n')

            print(f'Log file saved at: {logname}')

            if save:
                fname = os.path.join(self.output_dir, f'results.npz')
                np.savez(
                    fname, rt=rt, func0_calls=func0_calls, x0_len=x0_len,
                    permut=permut, x0=x0, cost0=cost0, xf=xf, costf=costf,
                    reduction_in_cost=reduction_in_cost, T0=T0, alpha=alpha,
                    epochs=epochs, N_per_epochs=N_per_epochs,
                    cooling_func=cooling_func, cost_hist=cost_hist
                )
                print(f'\nSummary data saved at: {fname}')
        finally:
            outputFile.close()

    @staticmethod
    def factorial_apprx(N):
        '''
            Find the approximate of the factorial of a large number!

            Parameters:
            -----------
            N: (int)
                Number for which the approximate factorial needs to be
                calculated

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


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    # Parse CLI arguments
    args = cli_parser()

    # Make output directory
    output_dir = make_output_dir(args.output_dir)

    # Set-up parameters for the Simulated Annealing Algorithm
    T0, alpha = args.T, args.alpha
    epochs, N_per_epochs = args.epochs, args.N_per_epochs
    SAVE, ext = args.SAVE, args.ext
    cfunc = args.cfunc

    # Initial solution
    tc_number = args.tc_number
    if args.tc_sym == 'sym':
        tc_fname = 'Choose_TC_Sym_NPZ.txt'
    elif args.tc_sym == 'asym':
        tc_fname = 'Choose_TC_Asym_NPZ.txt'

    # Read data off of standard library
    # loader = TestCaseLoader()
    loader = get_input_loader(tc_fname, False)
    tc_name = loader.get_test_case_name(tc_number)
    cost_matrix = loader.get_input_test_case(tc_number).get_cost_matrix()

    num_nodes = np.shape(cost_matrix)[0]
    print(f'\nNum of Nodes: {num_nodes}')
    np.random.seed()
    initial_soln = np.random.permutation(np.arange(num_nodes))

    # Set up Simulated Annealing Class
    optim_solution = SimpleSimulatedAnnealing(
        func0=get_total_cost, x0=initial_soln, T0=T0, alpha=alpha,
        epochs=epochs, N_per_epochs=N_per_epochs, cost_matrix=cost_matrix,
        cooling_func=cfunc, ext=ext, output_dir=output_dir
    )

    optim_solution.solver_summary(tc_name=tc_name, save=SAVE)
    optim_solution.plot_cost_hist(save=SAVE)

    # ---------------------------------------------------------------------
    # Generate random coordinates
    # initial_coords = Coordinate.random_coordinates_list(n=80, seed=0)

    # Set up Simulated Annealing Class
    # optim_solution = SimpleSimulatedAnnealing(
    #     func0=Coordinate.get_total_distance, x0=initial_coords, T0=T0,
    #     alpha=alpha, epochs=epochs, N=N_per_epochs
    # )

    # optim_solution.plot_cost_hist()
    # Coordinate.plot_solution(initial_coords, optim_solution.xf)

    # ---------------------------------------------------------------------
