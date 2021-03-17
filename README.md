# Museum-Path-Optimization

- [Museum-Path-Optimization](#museum-path-optimization)
  - [Group Members](#group-members)
  - [Description](#description)
    - [Data Input](#data-input)
    - [Branch and Bound](#branch-and-bound)
    - [Simulated Annealing](#simulated-annealing)
  - [Project Tree](#project-tree)

## Group Members
*In alphabetical order:*

Apurva Kulkarni, Arsh Khan, Harshal Kataria, K T Prajwal Prathiksh, Miloni Atal, Mridul Agarwal, Patel Joy Pravin Kumar, Nakul Randad, Souvik Kumar Dolui, Umang Goel

## Description
Contains code meant to optimize the route for a tourist visiting the Louvre Museum, such that the satisfaction level is maximised by visiting all/select exhibits in a single working day. 

This repository represents the work done as part of the course project for AE - 755: Optimization for Engineering Design *(Spring 2020)*, [Prof. Abhijit Gogulapati](https://www.aero.iitb.ac.in/home/people/faculty/abhijit), Indian Institute of Technology Bombay.

Instructions on running specific algorithms are mentioned below:

*Note: All of the commands mentioned below support CLI. Use the argument `-h` for help in each case.*

### Data Input
To generate and store the cost matrices of all the test cases, do the following from root:
```
$ python code/data_input/base_input.py
```

### Branch and Bound
To run the branch and bound algorithm, do the following from root:
```
$ python code/branch_and_bound/time_opti.py
```
Run the following to get all the command-line arguments:
```
$ python code/branch_and_bound/time_opti.py -h
```

### Simulated Annealing

**Simple Algorithm**

To run the simple simulated annealing algorithm, do the following from root:
```
$ python code/simulated_annealing/simple_simulated_annealing.py
```

**Complex Algorithm**

To run the complex simulated annealing algorithm, do the following from root:
```
$ python code/simulated_annealing/complex_simulated_annealing.py
```

**Automator**

To run the automator file, do the following from root:
```
$ python code\simulated_annealing\automate.py
```

## Project Tree
```
.
├───code
│   ├───ant_colony
│   ├───branch_and_bound
│   ├───data_input
│   ├───genetic
│   └───simulated_annealing
├───data
│   ├───cost_matrices
│   └───Symmetric_TSPLIB
├───output
└───reports
```
