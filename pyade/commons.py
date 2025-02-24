import numpy as np
from typing import Callable, Union, List, Tuple, Any


def keep_bounds(population: np.ndarray,
                bounds: np.ndarray) -> np.ndarray:
    """
    Constrains the population to its proper limits.
    Any value outside its bounded ranged is clipped.
    :param population: Current population that may not be constrained.
    :type population: np.ndarray
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype np.ndarray
    :return: Population constrained within its bounds.
    """
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    return np.clip(population, minimum, maximum)


def init_population(population_size: int, individual_size: int,
                    bounds: Union[np.ndarray, list]) -> np.ndarray:
    """
    Creates a random population within its constrained bounds.
    :param population_size: Number of individuals desired in the population.
    :type population_size: int
    :param individual_size: Number of features/gens.
    :type individual_size: int
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype: np.ndarray
    :return: Initialized population.
    """

    population = np.random.randn(population_size, individual_size)
    return keep_bounds(population, bounds)


def apply_fitness(population: np.ndarray,
                  func: Callable[[np.ndarray], float],
                  opts: Any) -> np.ndarray:
    """
    Applies the given fitness function to each individual of the population.
    :param population: Population to apply the current fitness function.
    :type population: np.ndarray
    :param func: Function that is used to calculate the fitness.
    :type func: np.ndarray
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :rtype np.ndarray
    :return: Numpy array of fitness for each individual.
    """
    if opts is None:
        return np.array([func(individual) for individual in population])
    else:
        return np.array([func(individual, opts) for individual in population])


def __parents_choice(population: np.ndarray, n_parents: int) -> np.ndarray:
    pob_size = population.shape[0]
    choices = np.indices((pob_size, pob_size))[1]
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    choices = choices[mask].reshape(pob_size, pob_size - 1)
    parents = np.array([np.random.choice(row, n_parents, replace=False) for row in choices])

    return parents


def binary_mutation(population: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    """
    Calculate the binary mutation of the population. For each individual (n),
    3 random parents (x,y,z) are selected. The parents are guaranteed to not
    be in the same position than the original. New individual are created by
    n = z + F * (x-y)
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents = __parents_choice(population, 3)
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 0]] - population[parents[:, 1]])
    mutated += population[parents[:, 2]]

    return keep_bounds(mutated, bounds)


def current_to_best_2_binary_mutation(population: np.ndarray,
                                      population_fitness: np.ndarray,
                                      f: Union[int, float],
                                      bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to best/2/bin" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 3:
        return population

    # 1. We find the best parent
    best_index = np.argmin(population_fitness)

    # 2. We choose two random parents
    parents = __parents_choice(population, 2)
    mutated = population + f * (population[best_index] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)


def current_to_pbest_mutation(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to p-best" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float, np.ndarray]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))

    p_best = np.array(p_best)
    # 2. We choose two random parents
    parents = __parents_choice(population, 2)
    mutated = population + f * (population[p_best] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)


def current_to_rand_1_mutation(population: np.ndarray,
                              population_fitness: np.ndarray,
                              k: List[float],
                              f: List[float],
                              bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to rand/1" mutation. This is
    U_{i, G} = X_{i, G} + K * (X_{r1, G} - X_{i, G} + F * (X_{r2. G} - X_{r3, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: Union[int, float]
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents = __parents_choice(population, 3)
    # 2. Apply the formula to each set of parents
    mutated = k * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])

    return keep_bounds(mutated, bounds)


def current_to_pbest_weighted_mutation(population: np.ndarray,
                                       population_fitness: np.ndarray,
                                       f: np.ndarray,
                                       f_w: np.ndarray,
                                       p: float,
                                       bounds: np.ndarray) -> np.ndarray:
    """
    Calculates the mutation of the entire population based on the
    "current to p-best weighted" mutation. This is
    V_{i, G} = X_{i, G} + F_w * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param population_fitness: Fitness of the given population
    :type population_fitness: np.ndarray
    :param f: Parameter of control of the mutation. Must be in [0, 2].
    :type f: np.ndarray
    :param f_w: NumPy Array with the weighted version of the mutation array
    :type f_w: np.ndarray
    :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
    :type p: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype: np.ndarray
    :return: Mutated population
    """
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    best_index = np.argsort(population_fitness)[:max(2, round(p*len(population)))]

    p_best = np.random.choice(best_index, len(population))
    # 2. We choose two random parents
    parents = __parents_choice(population, 2)
    mutated = population + f_w * (population[p_best] - population)
    mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])

    return keep_bounds(mutated, bounds)


def crossover(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
    """
    Crosses gens from individuals of the last generation and the mutated ones
    based on the crossover rate. Binary crossover
    :param population: Previous generation population.
    :type population: np.ndarray
    :param mutated: Mutated population.
    :type population: np.ndarray
    :param cr: Crossover rate. Must be in [0,1].
    :type population: Union[int, float]
    :rtype: np.ndarray
    :return: Current generation population.
    """
    chosen = np.random.rand(*population.shape)
    j_rand = np.random.randint(0, population.shape[1])
    chosen[j_rand::population.shape[1]] = 0
    return np.where(chosen <= cr, mutated, population)

def exponential_crossover(population: np.ndarray, mutated: np.ndarray,
                          cr: Union[int, float]) -> np.ndarray:
    """
        Crosses gens from individuals of the last generation and the mutated ones
        based on the crossover rate. Exponential crossover.
        :param population: Previous generation population.
        :type population: np.ndarray
        :param mutated: Mutated population.
        :type population: np.ndarray
        :param cr: Crossover rate. Must be in [0,1].
        :type population: Union[int, float]
        :rtype: np.ndarray
        :return: Current generation population.
    """
    if type(cr) is int or float:
        cr = np.array([cr] * len(population))
    else:
        cr = cr.flatten()

    def __exponential_crossover_1(x: np.ndarray, y: np.ndarray, cr: Union[int, float]) -> np.ndarray:
        z = x.copy()
        n = len(x)
        k = np.random.randint(0, n)
        j = k
        l = 0
        while True:
            z[j] = y[j]
            j = (j + 1) % n
            l += 1
            if np.random.randn() >= cr or l == n:
                return z

    return np.array([__exponential_crossover_1(population[i], mutated[i], cr.flatten()[i]) for i in range(len(population))])


def selection(population: np.ndarray, new_population: np.ndarray,
              fitness: np.ndarray, new_fitness: np.ndarray,
              return_indexes: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Selects the best individuals based on their fitness.
    :param population: Last generation population.
    :type population: np.ndarray
    :param new_population: Current generation population.
    :type new_population: np.ndarray
    :param fitness: Last generation fitness.
    :type fitness: np.ndarray
    :param new_fitness: Current generation fitness
    :param return_indexes: When active the function also returns the individual indexes that have been modified
    :type return_indexes: bool
    :rtype: ndarray
    :return: The selection of the best of previous generation
     and mutated individual for the entire population and optionally, the indexes changed
    """
    indexes = np.where(fitness > new_fitness)[0]
    population[indexes] = new_population[indexes]
    if return_indexes:
        return population, indexes
    else:
        return population
