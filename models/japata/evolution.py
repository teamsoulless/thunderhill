import numpy as np
from scipy.spatial import distance
import models


class Individual(object):
    def __init__(self, n_params, mu, sigma, mutation_prob):
        self._size = n_params
        self._distribution = (mu, sigma)
        self._mutation_prob = mutation_prob
        self.weights = np.random.normal(mu, sigma, n_params)
        self.fitness = None

    # mutate
    def __invert__(self):
        for idx in range(self._size):
            if np.random.uniform() < self._mutation_prob:
                self.weights[idx] += np.random.normal(*self._distribution)
        self.fitness = None

    # mate
    def __mul__(self, other):
        assert isinstance(other, Individual)
        assert self._size == other._size, 'Individuals have different sizes.'

        cut_pts = np.random.randint(0, self._size, 2)
        cut1 = min(cut_pts)
        cut2 = max(cut_pts)

        tmp = np.copy(self.weights[cut1:cut2])
        self.weights[cut1:cut2] = other.weights[cut1:cut2]
        other.weights[cut1:cut2] = tmp
        self.fitness = None

    # compare individuals with `max`
    def __gt__(self, other):
        assert self.fitness is not None and other.fitness is not None, 'Fitness is not initialized.'

        # if they have the same fitness, choose the one with the smallest weights
        if self.fitness == other.fitness:
            self_mean = np.mean(self.weights)
            other_mean = np.mean(other.weights)
            return self_mean < other_mean
        return self.fitness > other.fitness

    def evaluate_fitness(self, paths):
        # TODO
        pass

    def set_mutation_prob(self, mutation_prob):
        self._mutation_prob = mutation_prob

    def reset_weights(self):
        self.weights = np.random.normal(*self._distribution, self._size)


class Population(object):
    def __init__(self, size, base_individual, mate_prob):
        self._size = size
        self.mate_prob = mate_prob
        self.population = [base_individual.reset_weights() for _ in range(size)]

    def tournament(self, size):
        new_population = []
        for _ in range(self._size):
            new_population.append(max(
                np.random.choice(self.population, size)
            ))
        self.population = new_population

    def evolve(self, method, paths):
        assert method in ('and', 'or'), 'Method must be "and" or "or".'

        if method == 'and':
            self._random_mating(self.mate_prob)
            self._random_mutation()
        else:
            if np.random.uniform() < 0.5:
                self._random_mutation()
            else:
                self._random_mating(self.mate_prob)

        for individual in self.population:
            if individual.fitness is None:
                individual.evaluate_fitness(paths)

    def _random_mating(self, prob):
        for individual, mate in self._random_mate_generator(self.population):
            if np.random.uniform() < prob:
                individual * mate

    def _random_mutation(self):
        for individual in self.population:
            ~individual

    @staticmethod
    def _random_mate_generator(array):
        indexes = np.arange(len(array))
        for ind in indexes:
            mate = np.random.choice(np.delete(indexes, ind))
            yield array[ind], array[mate]


def get_layer_shapes(model):
    return [layer.shape for layer in model.get_weights()]


def construct_layers(weights, layer_shapes):
    layer_shape_sum = sum([np.prod(layer) for layer in layer_shapes])
    assert layer_shape_sum == weights.shape[0], 'Dimension mismatch.'

    reshaped_weights = []
    start_idx = 0
    for idx in range(len(layer_shapes)):
        shape = layer_shapes[idx]
        end_idx = start_idx + np.prod(shape)
        reshaped_weights.append(
            weights[start_idx:end_idx].reshape(shape)
          )
    return reshaped_weights


def fitness_function(array):
    pairwise_euclidean_distances = distance.pdist(array, 'euclidean')


def evaluate_fitness(individual, model, generator, fitness_func):
    # Get weights from individual and transform into layers
    # Load weights into model
    # Get output from the model
    # Evaluate the fitness and store in the individual
    pass


if __name__ == '__main__':
    model = models.cg23()

    individual = Individual(
        n_params=model.count_params(),
        mu=0,
        sigma=1.5,
        mutation_prob=0.05
      )
    population = Population(
        size=100,
        base_individual=individual,
        mate_prob=0.8
      )

    population.tournament(size=10)
    population.evolve(method='and', paths=None)
