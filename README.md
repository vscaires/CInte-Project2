# Computational Intelligence 

## Single Objective Optimization Problem

To run the program, there are some constants that can be changed which are
```python
    POPULATION = 40
    GENERATIONS = 250
    MAX_RUNS = 1
    CXPB = 0.9
    MUTPB = 0.3
```
To run more than one run, you have to change the MAX_RUNS to whatever number of runs you want to run.

If you want to run the program for a specific random seed, you need to specify it in the code
```python
    random.seed(Seed Number)

```

When the program is executed, there is this prompt to choose between the different datasets with a terminal input.
```bash
    ***********************************************
    *--- Single-Objective Optimization Problem ---*
    *    Choose One :                             *
    *         1 - Car Distances                   *
    *         2 - Car Costs                       *
    *         3 - Plane Distances                 *
    *         4 - Plane Costs                     *
    *                                             *
    *         0 - EXIT                            *
    ***********************************************
```


Then select the number of cities:
```bash
    ***********************************************
    *--- Single-Objective Optimization Problem ---*
    *    Number of Cities :                       *
    *         1 - 20                              *
    *         2 - 30                              *
    *         3 - 50                              *
    ***********************************************
```
And finally select if the program runs with Heuristics or not:

```bash
    ***********************************************
    *--- Single-Objective Optimization Problem ---*
    *    Choose One :                             *
    *         1 - Normal                          *
    *         2 - Heuristic                       *
    ***********************************************
```

After this, the program runs for all the generations and computes the best individual after. In the end, there is a plot of the best path and then the convergence curve of the best individual of each generation.
___
## Multi Objective Optimization Problem
Like the single objective program, there are also some constants declared in the beggining of the code to define some parameters for the algorithm. 
After running there is only one input to do which is the number of cities. 

```bash
    ***********************************************
    *--- Single-Objective Optimization Problem ---*
    *    Number of Cities :                       *
    *         1 - 20                              *
    *         2 - 30                              *
    *         3 - 50                              *
    ***********************************************
```
Then it will run the algorithm and produce the plot of the pareto front, the hypervolume evolution curve and the best path.