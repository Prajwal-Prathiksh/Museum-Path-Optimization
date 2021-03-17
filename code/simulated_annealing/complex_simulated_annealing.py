###########################################################################
# Imports
###########################################################################
# Standard library imports
import argparse
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())  # Insert this when you have any local imports


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
        output_dir = os.path.join(OUTPUT_DIR, 'CSA')
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
        '--n', action='store', dest='n', type=int,
        default=50, help='Total number of exhibits in the Muesuem'
    )
    parser.add_argument(
        '--vel', action='store', dest='vel', type=float,
        default=1.0, help='Velocity of the tourist (in m/s)'
    )
    parser.add_argument(
        '--t-max', action='store', dest='T_max', type=float,
        default=20,
        help='Maximum time the tourist can spend in the museum (in s)'
    )
    parser.add_argument(
        '--delta', action='store', dest='delta', type=int,
        default=10, help='Number of iterations after which solution is shaken'
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
        '--k', action='store', dest='k', type=float,
        default=0.6, help='Probability of increasing the exhibits visited'
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
    @function_calls
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
            np.random.seed(seed)

        coords = []
        for i in range(n):
            coords.append(Coordinate(np.random.uniform(), np.random.uniform()))

        return coords

    @staticmethod
    def get_feasible_solution(
        n, velocity, T_max, seed=None, low=1.0, high=5.0
    ):
        '''
            Generate an initial feasible solution to optimize the path taken by
            a tourist in a Muesuem and a corresponding Satisfaction array for
            each of the exhibit in the Muesuem.

            Parameters:
            -----------
            n: (int)
                Total number of exhibits in the Muesuem
            velocity: (float):
                Velocity of the tourist (in m/s)
            T_max: (int)
                Maximum time the tourist can spend in the museum (in s)
            seed: (int), default=None
                Reseed random number generator
            low & high: (int)
                Return random integers from the "discrete uniform"
                distribution of the specified dtype in the "half-open"
                interval [low, high). If high is None (the default), then
                results are from [0, low).

            Returns:
            --------
            initial_coords: (List)
                List of Coordinate instances
            initial_solution: (List)
                List of Coordinate indices corresponding to a feasible solution
            S: (List)
                Array of satisfaction level of each exhibit in the Muesuem
            loc_bar: (int)
                Number of exhibits that are to be visited by the tourist
        '''
        initial_coords = Coordinate.random_coordinates_list(n, seed=seed)

        if seed is None:
            np.random.seed()
        else:
            np.random.seed(seed)

        S, initial_solution = [], []
        for i in range(n):
            S.append(np.random.randint(low=low, high=high,))
            initial_solution.append(i)

        loc_bar = np.random.randint(1, n)

        while(
            not Coordinate.constraints(
                initial_solution[:loc_bar], initial_coords, velocity, T_max
            )
        ):
            np.random.shuffle(initial_solution)
            loc_bar = np.random.randint(1, n)

        return initial_coords, initial_solution, S, loc_bar

    @staticmethod
    @function_calls
    def satisfaction(x, S):
        '''
            Finds the satisfaction level of Tourist

            Parameters:
            -----------
            x: (List)
                List of Indices
            S: (List)
                Satisfaction array
            Returns:
            --------
            Satisfaction level
        '''
        satisfaction = 0
        for i in x:
            satisfaction += S[i]
        return satisfaction

    @staticmethod
    @function_calls
    def time_taken(x, coords, velocity):
        '''
            Time to complete tour

            Parameters:
            -----------
            x: (List)
                List of Indices
            coords: (List)
                List of Coordinates
            velocity: (float)
                Velocity of tourist (m/s)

            Returns:
            --------
            Time to complete tour
        '''
        dist = 0.0
        for i in range(0, len(x) - 1, 1):
            # Distance between successive exhibits are added
            dist += Coordinate.get_distance(coords[x[i]], coords[x[(i + 1)]])

        # Distance between the first and the last exhibits is added
        dist += Coordinate.get_distance(coords[x[0]], coords[x[-1]])
        return (dist / velocity)

    @staticmethod
    @function_calls
    def constraints(x, coords, velocity, T_max):
        '''
            Checks the constraints

            Parameters:
            -----------
            x: (List)
                List of Indices
            coords: (List)
                List of Coordinates
            velocity: (float)
                Velocity of tourist (m/s)
            T_max: (int)
                Maximum time limit (secs)
            Returns:
            --------
            Boolean value of whether constraints are satisfied or not
        '''
        return (Coordinate.time_taken(x, coords, velocity) < T_max)

    @staticmethod
    def plot_solution(
        initial_coords, initial_solution, loc_bar, final_solution,
        final_loc_bar, S, output_dir, ext='', save=False,
    ):
        '''
            Plots the inital and the optimized solution in a convinient format.

            Parameters:
            -----------
            initial_coords: (List)
                Inital list of Coordinate instances
            initial_solution: (List)
                List of Indices
            loc_bar: (int)
                Initial location of bar, number of exhibits visited
            final_solution: (List)
                List of final indices
            final_loc_bar: (int)
                Final location of bar, number of exhibits visited
            S: (List)
                Array of satisfaction level of each exhibit in the Muesuem
            output_dir: (string)
                Absolute path of the output directory
            ext: (string), default=''
                Add a prefix to the plot before saving it
            save = (Boolean), default=True
                If True, saves the plot

            Returns:
            --------
            Plot
        '''
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cmap_n_digit = 3  # round(max(S)*0.5)

        # Initial Solution
        for i in range(0, loc_bar - 1):
            first = initial_coords[initial_solution[i]]
            second = initial_coords[initial_solution[i + 1]]
            ax1.plot(
                [first.x, second.x], [first.y, second.y], 'k--', linewidth=0.65
            )

        ax1.plot(
            [initial_coords[initial_solution[0]].x,
             initial_coords[initial_solution[loc_bar - 1]].x],
            [initial_coords[initial_solution[0]].y,
             initial_coords[initial_solution[loc_bar - 1]].y],
            'k--', linewidth=0.65
        )

        coord_x, coord_y = [], []
        for c in initial_coords:
            coord_x.append(c.x)
            coord_y.append(c.y)

        ax1.scatter(
            coord_x, coord_y,
            c=S, lw=0.1, cmap=plt.cm.get_cmap('jet', cmap_n_digit)
        )

        # Optimized Solution
        for i in range(0, final_loc_bar - 1):
            first = initial_coords[final_solution[i]]
            second = initial_coords[final_solution[i + 1]]
            ax2.plot(
                [first.x, second.x], [first.y, second.y], 'k--', linewidth=0.65
            )

        ax2.plot(
            [initial_coords[final_solution[0]].x,
             initial_coords[final_solution[final_loc_bar - 1]].x],
            [initial_coords[final_solution[0]].y,
             initial_coords[final_solution[final_loc_bar - 1]].y],
            'k--', linewidth=0.65
        )

        im = ax2.scatter(
            coord_x, coord_y,
            c=S, lw=0.1, cmap=plt.cm.get_cmap('jet', cmap_n_digit)
        )

        old_cost = round(Coordinate.satisfaction(
            initial_solution[:loc_bar], S), 2)
        new_cost = round(Coordinate.satisfaction(
            final_solution[:final_loc_bar], S), 2)

        ax1.title.set_text(f'Initial Solution | Cost = {old_cost}')
        ax2.title.set_text(f'Optimized Solution | Cost = {new_cost}')

        fig.colorbar(
            im, ax=[ax1, ax2], label=r'Satisfaction Level $\rightarrow$'
        )

        if save:
            fname = os.path.join(
                output_dir, f'{ext}CSA_optimized_solution.png'
            )
            plt.savefig(fname, dpi=400, bbox_inches='tight')
            print(f'\nPlot saved at: {fname}')


class ComplexSimulatedAnnealing:
    '''
        Simulated Annealing Class
        ---------------------------------
        A complex version of the Simulated Annealing algorithm.

        Parameters:
        -----------
        func0: (Function)
            Cost function to be evaluated
        check_constraints: (Function)
            Checks that the function obeys the constraints
        coords: (List)
            List of Coordinate instances
        x0: (Array):
            Initial solution
        loc_bar: (int):
            Location of bar separating visited and unvisited exhibit
        velocity: (float):
            Velocity of the tourist (in m/s)
        T_max: (int)
            Maximum time the tourist can spend in the museum (in s)
        S: (Array):
            Satisfaction from visiting each exhibit
        T0: (float)
            Inital temperature
        alpha: (float)
            Cooling factor
        epochs: (int)
            Number of epochs
        N_per_epochs: (int)
            Number of iterations per epoch
        delta: (int)
            Number of iterations after which solution is shaken
        output_dir: (string)
            Absolute path of the output directory
        cooling_func: (string), default=simple
            Type of cooling function to be used
            Choose from: ['simp', 'exp']
        ext: (string), default=''
            Add a prefix to the plots, summary_data and summary_log before
            saving it
    '''

    def __init__(
        self, func0, check_constraints, coords, x0, loc_bar, velocity, T_max,
        S, T0, alpha, epochs, N_per_epochs, delta, k, output_dir,
        cooling_func='simp', ext='', **kwargs
    ):
        '''
            Parameters:
            -----------
            func0: (Function)
                Cost function to be evaluated
            check_constraints: (Function)
                Checks that the function obeys the constraints
            coords: (List)
                List of Coordinate instances
            x0: (Array):
                Initial solution
            loc_bar: (int):
                Location of bar separating visited and unvisited exhibit
            velocity: (float):
                Velocity of the tourist (in m/s)
            T_max: (int)
                Maximum time the tourist can spend in the museum (in s)
            S: (Array):
                Satisfaction from visiting each exhibit
            T0: (float)
                Inital temperature
            alpha: (float)
                Cooling factor
            epochs: (int)
                Number of epochs
            N_per_epochs: (int)
                Number of iterations per epoch
            delta: (int)
                Number of iterations after which solution is shaken
            k: (float)
                Between (0-1), Probabilty that the number of exhibits visited
                increases
            output_dir: (string)
                Absolute path of the output directory
            cooling_func: (string), default=simple
                Type of cooling function to be used
                Choose from: ['simp', 'exp']
            ext: (string), default=''
                Add a prefix to the plots, summary_data and summary_log before
                saving it
        '''
        self.output_dir = output_dir
        # Initialize
        self.func0 = func0
        self.check_constraints = check_constraints
        self.cost0 = round(self.func0(x0[:loc_bar], S), 3)

        # Set number of function calls to zero
        self.func0_calls = 0
        self.check_constraints_calls = 0

        self.coords = coords
        self.S = S
        self.velocity = velocity
        self.T_max = T_max
        self.x0 = x0.copy()
        self.loc_bar = loc_bar
        self.T0 = T0
        self.alpha = alpha
        self.epochs = epochs
        self.N_per_epochs = N_per_epochs
        self.delta = delta
        self.k = k
        self.ext = ext

        # Print initial conditions
        print(f'\nTotal Number of Epochs: {self.epochs}')
        print(f'Initial Cost: {self.cost0} | ' +
              f'Inital Exhibits visited: {self.loc_bar}\n'
              )

        # Cooling function
        self.cooling_funcs_dict = dict(
            simp=self.simple_cooling_func, exp=self.exponential_cooling_func
        )
        self.cooling_func = self.cooling_funcs_dict[cooling_func]

        self.len_x0 = len(x0)

        # Run Algorithm
        self.xf, self.final_loc_bar, self.cost_hist, self.rt = \
            self.run_algorithm()
        self.costf = round(self.cost_hist[-1], 3)

        self.increase_in_cost = round(
            np.abs(1 - self.cost0 / self.costf) * 100, 3
        )

        self.func0_calls = self.func0.calls
        self.check_constraints_calls = self.check_constraints.calls

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
                self.output_dir, f'{ext}CSA_cost_hist.png'
            )
            plt.savefig(
                fname, dpi=400, bbox_inches='tight'
            )
            print(f'\nPlot saved at: {fname}')
        plt.close()

    @staticmethod
    def apply_swap(x, loc_bar):
        '''
            Swaps two elements of array which are being visited

            Parameters:
            -----------
            x: (List)
                List of Indices
            loc_bar: (int)
                Location of bar, Indicating the number of exhibits being
                visited
            Returns:
            --------
            Swapped solution
        '''
        if(loc_bar < 2):
            return x
        x0 = x.copy()
        e1, e2 = np.random.randint(0, loc_bar, size=2)
        temp = x0[e1]
        x0[e1] = x0[e2]
        x0[e2] = temp
        return x0

    @staticmethod
    def apply_shake(x, loc_bar):
        '''
            Swaps two elements of array one being visited, one not being
            visited.

            Parameters:
            -----------
            x: (List)
                List of Indices
            loc_bar: (int)
                Location of bar, Indicating the number of exhibits being
                visited
            Returns:
            --------
            Swapped solution
        '''
        x0 = x.copy()
        if(loc_bar == len(x0)):
            return x0

        e1 = np.random.randint(0, loc_bar)
        e2 = np.random.randint(loc_bar, len(x0))
        temp = x0[e1]
        x0[e1] = x0[e2]
        x0[e2] = temp
        return x0

    @staticmethod
    def modify_nodes(x, loc_bar, k):
        '''
            Increases or decreases the exhibits visited

            Parameters:
            -----------
            x: (List)
                List of Indices
            loc_bar: (int)
                Location of bar, Indicating the number of exhibits being
                visited
            k: (float)
                Probabilty that the location of bar increases
            Returns:
            --------
            New location of bar
        '''
        if(loc_bar == 1):
            loc_bar += 1
        elif(loc_bar == len(x)):
            loc_bar -= 1
        else:
            if(np.random.random() < k):
                loc_bar += 1
            else:
                loc_bar -= 1
        return loc_bar

    def run_algorithm(self):
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
        coords = self.coords
        loc_bar = self.loc_bar
        velocity = self.velocity
        T_max = self.T_max
        S = self.S
        T = self.T0
        k = self.k
        cost = self.func0(x[:loc_bar], S)

        apply_swap = ComplexSimulatedAnnealing.apply_swap
        apply_shake = ComplexSimulatedAnnealing.apply_shake
        modify_nodes = ComplexSimulatedAnnealing.modify_nodes
        time_taken = Coordinate.time_taken

        x_best = x
        cost_best = cost
        loc_bar_best = loc_bar

        i = 0
        j = 0
        np.random.seed()  # Reset seed

        cost_hist = []
        tic = time.monotonic()

        for epoch in range(self.epochs):
            # Store history of cost
            cost_hist.append(cost)

            tmp_constr = self.check_constraints(
                x[:loc_bar], coords, velocity, T_max
            )
            tmp_time = round(time_taken(x[:loc_bar], coords, velocity), 1)
            msg = f'Epoch: {epoch} | Cost = {round(cost, 1)} | ' +\
                f'Exhibits visited = {loc_bar} | t = {tmp_time} | ' +\
                f't_max = {T_max} | Constraints: {tmp_constr}'
            print(msg, end='\r')

            T = self.cooling_func(T, epoch)

            for iterator in range(self.N_per_epochs):
                loc_bar_new = modify_nodes(x, loc_bar, k)
                x_new = apply_shake(x, loc_bar_new)
                cost_new = self.func0(x_new[:loc_bar_new], S)

                if(
                    self.check_constraints(
                        x_new[:loc_bar_new], coords, velocity, T_max
                    )
                ):
                    if cost_new > cost:
                        x = x_new
                        loc_bar = loc_bar_new
                        cost = cost_new
                        if cost_new > cost_best:
                            x_best = x_new
                            loc_bar_best = loc_bar_new
                            cost_best = cost_new
                            i = 0
                        else:
                            i += 1
                    else:

                        if np.random.uniform() < np.exp((cost_new - cost) / T):
                            x = x_new
                            loc_bar = loc_bar_new
                            cost = cost_new
                else:
                    j += 1

                if(j >= self.delta):
                    x_new = apply_swap(x, loc_bar)
                    if(
                        self.check_constraints(
                            x_new[:loc_bar], coords, velocity, T_max
                        )
                    ):
                        x = x_new
                        cost = self.func0(x[:loc_bar], S)

        toc = time.monotonic()
        rt = toc - tic

        cost_hist = np.array(cost_hist)
        return x, loc_bar, cost_hist, round(rt, 3)

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
            logname = os.path.join(
                self.output_dir, 'CSA_solver_summary.log'
            )
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
            c_calls = self.check_constraints_calls
            printing(f'Number of constraint function calls: {c_calls}')

            x0_len = len(self.x0)
            printing(f'\nTotal number of nodes: {x0_len}')
            permut = ComplexSimulatedAnnealing.num_permutations_approx(x0_len)
            printing(f'Total number of permutations: {permut}')

            x0, xf = self.x0, self.xf
            cost0, costf = self.cost0, self.costf
            printing(
                f'\nInitial Cost: {cost0} ---> Optimized Cost: {costf}'
            )
            increase_in_cost = self.increase_in_cost
            printing(f'Increase in cost (in %): {increase_in_cost} %')

            T0, alpha, delta, k = self.T0, self.alpha, self.delta, self.k
            epochs = self.epochs
            N_per_epochs = self.N_per_epochs
            vel, T_max = self.velocity, self.T_max
            cooling_func = self.cooling_func.__name__
            printing(f'\nT0 = {T0} | alpha = {alpha}')
            printing(f'delta = {delta} | k = {k}')
            printing(f'epochs = {epochs} | iter/epoch = {N_per_epochs}')
            printing(f'Velocity = {vel} | T_max = {T_max}')
            printing(f'Cooling Function: {cooling_func}()')

            printing('\n===================================================\n')

            print(f'Log file saved at: {logname}')

            if save:
                fname = os.path.join(self.output_dir, f'results.npz')
                np.savez(
                    fname, rt=rt, func0_calls=func0_calls, x0_len=x0_len,
                    permut=permut, x0=x0, cost0=cost0, xf=xf, costf=costf,
                    increase_in_cost=increase_in_cost, T0=T0, alpha=alpha,
                    delta=delta, epochs=epochs, N_per_epochs=N_per_epochs,
                    cooling_func=cooling_func, vel=vel, T_max=T_max,
                    cost_hist=self.cost_hist
                )
                print(f'\nSummary data saved at: {fname}')
        finally:
            outputFile.close()

    @staticmethod
    def num_permutations_approx(N):
        '''
            Find the approximate of the total number of permutations for the
            tourist problem.

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
        tmp = list(range(1, N + 1))
        for i in range(N):
            tmp[i] = np.math.factorial(tmp[i])
        P = sum(tmp)
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

    # Generate random coordinates, initial solution that satifies constraints
    n = args.n
    velocity, T_max, delta, k = args.vel, args.T_max, args.delta, args.k

    feasible_solution = Coordinate.get_feasible_solution(
        n, velocity, T_max, seed=1, low=0, high=11
    )

    initial_coords, initial_solution, S, loc_bar = feasible_solution

    # Set-up parameters for the Simulated Annealing Algorithm
    T0, alpha, k = args.T, args.alpha, args.k
    epochs, N_per_epochs = args.epochs, args.N_per_epochs

    SAVE, ext = args.SAVE, args.ext
    cfunc = args.cfunc

    # Set up Simulated Annealing Class
    optim_solution = ComplexSimulatedAnnealing(
        func0=Coordinate.satisfaction,
        check_constraints=Coordinate.constraints,
        coords=initial_coords,
        x0=initial_solution,
        loc_bar=loc_bar,
        velocity=velocity,
        T_max=T_max,
        S=S,
        T0=T0,
        alpha=alpha,
        epochs=epochs,
        N_per_epochs=N_per_epochs,
        delta=delta,
        k=k,
        ext=ext,
        output_dir=output_dir
    )
    optim_solution.solver_summary(save=SAVE)
    optim_solution.plot_cost_hist(save=SAVE)
    Coordinate.plot_solution(
        initial_coords, initial_solution, loc_bar, optim_solution.xf,
        optim_solution.final_loc_bar, S, ext=ext, save=SAVE,
        output_dir=output_dir
    )
