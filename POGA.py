# ---------------------------------------------------------------------------------------------------------------
#
#                   POGA: Price Optimiaztion via Genetic Algorithm
#
#                             A. Rusinko  (3/24/2019) v1.0
# ---------------------------------------------------------------------------------------------------------------
import sys
import numpy
from matplotlib import pyplot

# Genetic Algorithm imported frrom GitHub
# author: Ahmed Gad
# https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/GA.py
# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
# Mutation Function modified by ARusinko
from GA import GeneticAlgorithm as GA

__author__ = 'ARusinko'

##################################
#       GLOBALS
##################################
FLOOR_PENALTY   = 10000.0
CEILING_PENALTY = 0.1
PRICE_PENALTY   = 10000.0

##################################
#       FUNCTIONS
##################################
def compute_fitness(guess, reference, floor, ceiling, quantities):
    """
    function: compute_fitness
    Compute fitness score based on deviations of guess from reference prices.
    Input:   guess      -- list of price guesses coming from GA
             reference  -- list of original guess prices
             floor      -- list of floor prices (minimum, hard)
             ceiling    -- list of ceiling prices (maximum, soft)
             quantities -- list of price break quantities
    Output: Return fitness score
    """
    global FLOOR_PENALTY, CEILING_PENALTY, PRICE_PENALTY

    # Initialize fitness score
    fitness = 0.00001

    # Euclidean distance of guess from reference
    ref_distance = [ (x-reference[i])*(x-reference[i]) for i,x in enumerate(guess)]
    fitness += sum(ref_distance)

    # Check for price decrease as volume increases --
    price_decrease = [PRICE_PENALTY if guess[i]<1.02*guess[i+1] else 0.0 for i in range(0,len(guess)-1)]
    fitness += sum(price_decrease)

    # Check for extended price increase as volume increases
    extprice_increase = [PRICE_PENALTY if guess[i]*quantities[i] > guess[i+1]*quantities[i+1] else 0.0 for i in range(0,len(guess)-1)]
    fitness += sum(extprice_increase)

    # Price below floor -- assign FLOOR_PENALTY per violation
    below_floor = [ FLOOR_PENALTY if x<floor[i] else 0.0 for i,x in enumerate(guess)]
    fitness += sum(below_floor)

    # Price above ceiling -- assign CEILING_PENALTY per violation
    above_ceiling = [ CEILING_PENALTY if x>ceiling[i] else 0.0 for i,x in enumerate(guess)]
    fitness += sum(above_ceiling)

    # Return inverse of fitness so that the smallest fitness value is rewarded with max value
    return round((1.0 / fitness), 5)

# ----------------------------------------------------------------------------------------------------
#		                                    MAIN Code
# ----------------------------------------------------------------------------------------------------
def main(argv):
    """
    function: main
    Main routine for computing price updates.
    """

    # Set to True if display of optimal trajectory is desired
    show_trajectory = False

    # switch between test input sets by changing inputval
    inputval = 2

    # Input1 for POGA
    if inputval == 1:
        guess      = [1.15, 0.9, 0.79, 0.53]
        floor      = [0.65, 0.65, 0.65, 0.65]
        ceiling    = [1.1, 1.0, 0.9, 0.8]
        quantities = [1, 5, 10, 25]

    # Input2 for POGA
    elif inputval == 2:
        guess      = [1.15, 0.88, 0.89, 0.63, 0.6, 0.58, 0.59]
        floor      = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
        ceiling    = [1.1, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
        quantities = [1, 5, 10, 25, 50, 100, 250]

    # Basic GA control parameters
    sol_per_pop = 1000                          # solutions per population
    num_prices  = len(guess)                    # number of genes per chromosome
    num_parents_mating = 50                     # number of top individuals allowed to mate
    max_num_generations = 1000                      # max number of generations
    max_optimal = 5                             # stopping criteria for no progress

    # The population will have sol_per_pop chromosome where each chromosome has num_prices genes.
    pop_size = (sol_per_pop, num_prices)

    # Creating the initial population.
    new_population = numpy.zeros(pop_size)
    for i in range(0, sol_per_pop):
        new_weights       = numpy.random.uniform(low=0.85, high=1.15, size=(1, num_prices))
        new_chromosome    = new_weights * guess
        new_population[i] = new_chromosome

    # Optimize new prices
    best_outputs    = []
    current_optimal    = -99999.99
    optimal_generation = 0
    for generation in range(max_num_generations):
        # Measuring the fitness of each chromosome in the population.
        pop_fitness = []
        for i in range(0,sol_per_pop):
            pop_fitness.append(compute_fitness(new_population[i], guess, floor, ceiling, quantities))

        # The best result in the current iteration
        best_outputs.append(max(pop_fitness))
        if generation%5==0: print("Generation : ", generation, "\t\tBest result : ",max(pop_fitness) )

        # Update optimal value and number of generations that found it
        if max(pop_fitness) > current_optimal:
            current_optimal = max(pop_fitness)
            optimal_generation = 0
        else:
            optimal_generation += 1
            # Break if no progress found after max_optimal generations
            if optimal_generation == max_optimal and current_optimal>0.0001:
                break
            elif optimal_generation==int(0.1*max_num_generations) and current_optimal<0.00011:
                break

        # Break if last generation, no need to update new_population
        if generation == max_num_generations-1: break

        # Selecting the best parents in the population for mating.
        parents = GA.select_mating_pool(new_population, pop_fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_prices))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = GA.mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :]  = offspring_mutation

    # Identify the best solution obtained
    best_value     = max(pop_fitness)
    best_match_idx = pop_fitness.index(best_value)
    print("\nOriginal guess: ", guess)
    print("Best solution : ", numpy.round(new_population[best_match_idx, :],3))
    print("Optimal Fitness=", best_value,"in",generation,"generations")
    print("Floor values: ", floor)
    print("Ceiling values: ", ceiling)

    # Plot the solution space
    pyplot.plot(quantities, guess, color='blue', marker='o',linewidth=0.75, label='Initial')
    pyplot.plot(quantities, new_population[best_match_idx, :] , color='red', marker='o',linewidth=0.75,label='Optimal')
    pyplot.plot(quantities, ceiling , color='green', linestyle='dashed',linewidth=0.75,label='Ceiling')
    pyplot.plot(quantities, floor, color='black',label='Floor')

    # Build polygon of possible solution space
    sol_space_x = quantities + quantities[::-1]
    sol_space_y = [1.15*x for x in guess]
    sol_space_y.extend([0.85*x for x in guess[::-1]])
    pyplot.fill(sol_space_x, sol_space_y, 'b', alpha=0.1 )

    # Decorate plot
    pyplot.title("Optimized Prices")
    pyplot.xlabel("Quantity Purchased")
    pyplot.ylabel("Price ($)")
    pyplot.legend()
    pyplot.show()

    # Plot the trajectory to optimal value
    if show_trajectory == True:
        pyplot.plot(best_outputs)
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Fitness")
        pyplot.title("Trajectory to Optimal Value")
        pyplot.show()

    pass

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EXECUTE Main
if __name__ == "__main__":
    main(sys.argv)
    sys.exit()