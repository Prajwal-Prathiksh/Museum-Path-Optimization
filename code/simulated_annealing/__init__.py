###########################################################################
# Imports
###########################################################################
# Standard library imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt


###########################################################################
# Code
###########################################################################
OUTPUT_DIR = os.path.join(
    os.getcwd(), 'output', 'figures', 'simulated_annealing'
)


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
    def time(x, coords, velocity):
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
        for i in range(0,len(x)-1,1):
            # Distance between successive exhibits are added
            dist += Coordinate.get_distance(coords[x[i]], coords[x[(i+1)]])

        # Distance between the first and the last exhibits is added
        dist += Coordinate.get_distance(coords[x[0]], coords[x[-1]])
        return (dist/velocity)
    
    @staticmethod
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
        return (Coordinate.time(x,coords,velocity) < T_max)

    
    @staticmethod
    def plot_solution(initial_coords,initial_solution,loc_bar, final_solution, final_loc_bar,S):
        '''
            Plots the inital and the optimized solution in a convinient format.

            Parameters:
            -----------
            initial_coords: (List)
                Inital list of Coordinate classes
            initial_solution: (List)
                List of Indices
            loc_bar: (int)
                Initial location of bar, number of exhibits visited
            final_solution: (List)
                List of final indices
            final_loc_bar: (int)
                Final location of bar, number of exhibits visited
            

            Returns:
            --------
            Plot
        '''
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Initial Solution
        for i in range(0,loc_bar-1):
            first = initial_coords[initial_solution[i]]
            second = initial_coords[initial_solution[i+1]]
            ax1.plot([first.x, second.x], [first.y, second.y], 'b')
        ax1.plot([initial_coords[initial_solution[0]].x, initial_coords[initial_solution[-1]].x],
                 [initial_coords[initial_solution[0]].y, initial_coords[initial_solution[-1]].y], 'b')

        for c in initial_coords:
            ax1.plot(c.x, c.y, 'ro')

        # Optimized Solution
        for i in range(0,final_loc_bar-1):
            first = initial_coords[final_solution[i]]
            second = initial_coords[final_solution[i+1]]
            ax2.plot([first.x, second.x], [first.y, second.y], 'b')
        ax2.plot([initial_coords[final_solution[0]].x, initial_coords[final_solution[-1]].x],
                 [initial_coords[final_solution[0]].y, initial_coords[final_solution[-1]].y], 'b')

        for c in initial_coords:
            ax2.plot(c.x, c.y, 'ro')

        old_cost = round(Coordinate.satisfaction(initial_solution[:loc_bar], S), 2)
        new_cost = round(Coordinate.satisfaction(final_solution[:final_loc_bar], S), 2)

        ax1.title.set_text(f'Initial Solution | Cost = {old_cost}')
        ax2.title.set_text(f'Optimized Solution | Cost = {new_cost}')

        fname = os.path.join(OUTPUT_DIR, 'SA_optimized_solution.png')
        plt.savefig(fname, dpi=400, bbox_inches='tight')
        plt.show()


class SimulatedAnnealing:
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
            x0: (Array):
                Initial solution
            loc_bar: (int):
                Location of bar separating visited and unvisited exhibit
            velocity: (float):
                Velocity of tourist (m/sec)
            T_max: (int)
                Maximum time the tourist has (secs)
            S: (Array):
                Satisfaction from visiting each exhibit
            T0: (float)
                Inital temperature
            alpha: (float)
                Cooling factor
            epochs: (int)
                Number of epochs
            N: (int)
                Number of iterations per epoch
            delta: (int)
                Number after which solution is shaken
    '''

    def __init__(self, func0, check_constraints ,coords, x0, loc_bar, velocity, T_max, S , T0, alpha, epochs, N, delta):
        '''
            Parameters:
            -----------
            func0: (Function)
                Cost function to be evaluated
            check_constraints: (Function)
                Checks that the function obeys the constraints
            x0: (Array):
                Initial solution
            loc_bar: (int):
                Location of bar separating visited and unvisited exhibit
            velocity: (float):
                Velocity of tourist (m/sec)
            T_max: (int)
                Maximum time the tourist has (secs)
            S: (Array):
                Satisfaction from visiting each exhibit
            T0: (float)
                Inital temperature
            alpha: (float)
                Cooling factor
            epochs: (int)
                Number of epochs
            N: (int)
                Number of iterations per epoch
            delta: (int)
                Number after which solution is shaken
            
        '''
        self.func0 = func0
        self.check_constraints = check_constraints
        self.coords = coords
        self.S = S
        self.velocity = velocity
        self.T_max = T_max
        self.x0 = x0.copy()
        self.loc_bar = loc_bar
        self.T0 = T0
        self.alpha = alpha
        self.epochs = epochs
        self.N = N
        self.delta = delta
        self.len_x0 = len(x0)

        # Run Algorithm
        self.xf, self.final_loc_bar, self.cost_hist, self.rt = self.run_algorithm()

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

    def plot_cost_hist(self):
        '''
            Plot the history of cost of the objective function per epoch.

            Returns:
            --------
            Plot
        '''
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)

        x = np.arange(1, self.epochs + 1)
        ax1.plot(x, self.cost_hist)

        ax1.title.set_text(f'Cost vs Epoch | Runtime: {round(self.rt,1)} s')
        ax1.set(xlabel=r'Epochs $\rightarrow$', ylabel='Cost')

        fname = os.path.join(OUTPUT_DIR, 'SA_cost_hist.png')
        plt.savefig(fname, dpi=400, bbox_inches='tight')
        plt.show()
    
    def apply_swap(x,loc_bar):
        '''
            Swaps two elements of array which are being visited

            Parameters:
            -----------
            x: (List)
                List of Indices
            loc_bar: (int)
                Location of bar, Indicating the number of exhibits being visited
            Returns:
            --------
            Swapped solution
        '''
        if(loc_bar <2):
            return x
        x0 = x.copy()
        e1, e2 = np.random.randint(0, loc_bar, size=2)
        temp = x0[e1]
        x0[e1] = x0[e2]
        x0[e2] = temp   
        return x0

    def apply_shake(x,loc_bar):
        '''
            Swaps two elements of array one being visited, one not being visited

            Parameters:
            -----------
            x: (List)
                List of Indices
            loc_bar: (int)
                Location of bar, Indicating the number of exhibits being visited
            Returns:
            --------
            Swapped solution
        '''
        x0 = x.copy()
        if(loc_bar == len(x0)):
            return x0
        
        e1 =  np.random.randint(0, loc_bar)  
        e2 = np.random.randint(loc_bar,len(x0))
        temp = x0[e1]
        x0[e1] = x0[e2]
        x0[e2] = temp   
        return x0

    def modify_nodes(x,loc_bar):
        '''
            Increases or decreases the exhibits visited

            Parameters:
            -----------
            x: (List)
                List of Indices
            loc_bar: (int)
                Location of bar, Indicating the number of exhibits being visited
            Returns:
            --------
            New location of bar
        '''
        if(loc_bar==1):
            loc_bar += 1
        elif(loc_bar==len(x)):
            loc_bar -= 1
        else:
            if(np.random.random()<0.6):
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
        cost = self.func0(x[:loc_bar], S)

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

            print(f'Epoch: {epoch} | Cost = {round(cost, 3)} |  {loc_bar} | {self.check_constraints(x[:loc_bar], coords, velocity, T_max)} | {Coordinate.time(x[:loc_bar],coords,velocity)} | {T_max}')

            T = self.cooling_func(T)

            for iterator in range(self.N):
                loc_bar_new = SimulatedAnnealing.modify_nodes(x,loc_bar)
                x_new = SimulatedAnnealing.apply_shake(x, loc_bar_new)
                cost_new = self.func0(x_new[:loc_bar_new],S)
                
                if(self.check_constraints(x_new[:loc_bar_new], coords, velocity, T_max)):
                    if cost_new > cost:
                        x = x_new
                        loc_bar = loc_bar_new
                        cost = cost_new
                        if cost_new > cost_best :
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
                
                if(j>= self.delta):
                    x_new = SimulatedAnnealing.apply_swap(x, loc_bar)
                    if(self.check_constraints(x_new[:loc_bar], coords, velocity, T_max)):
                        x = x_new
                        cost = self.func0(x[:loc_bar],S)
                    
        toc = time.monotonic()
        rt = toc - tic

        cost_hist = np.array(cost_hist)
        return x, loc_bar, cost_hist, rt


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':

    # Generate random coordinates, initial solution that satifies constraints
    initial_coords = Coordinate.random_coordinates_list(50, seed=0)
    velocity = 1
    T_max = 20
    S = []
    initial_solution = []
    for i in range(len(initial_coords)):
        S.append(np.random.uniform())
        initial_solution.append(i)
    loc_bar = np.random.randint(1,len(initial_solution))
    while(not Coordinate.constraints(initial_solution[:loc_bar],initial_coords,velocity,T_max)):
        np.random.shuffle(initial_solution)
        loc_bar = np.random.randint(1,len(initial_solution))

    # Set-up parameters for the Simulated Annealing Algorithm
    T0, alpha, outer_N, inner_N = [4, 0.99, 1000, 100]

    # Set up Simulated Annealing Class
    optim_solution = SimulatedAnnealing(
        func0=Coordinate.satisfaction ,check_constraints=Coordinate.constraints,coords=initial_coords, x0=initial_solution, loc_bar= loc_bar, velocity=velocity ,T_max=T_max,S=S, T0=T0,
        alpha=alpha, epochs=outer_N, N=inner_N, delta=10
    )
    optim_solution.plot_cost_hist()
    Coordinate.plot_solution(initial_coords,initial_solution,loc_bar, optim_solution.xf, optim_solution.final_loc_bar,S)
