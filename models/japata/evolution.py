import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.spatial import distance
import models
import utils


class Individual(object):
    def __init__(self, model_generator, mu, sigma, mutation_prob, fitness_func):
        self.model = model_generator()
        self._model_generator = model_generator
        self._layer_shapes = [layer.shape for layer in self.model.get_weights()]
        self._size = self.model.count_params()
        self._distribution = (mu, sigma)
        self._mutation_prob = mutation_prob
        self._fitness_func = fitness_func
        self.weights = np.random.normal(mu, sigma, self._size)
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

    def evaluate_fitness(self, generator):
        self._update_model_weights()

        feature_vectors = None
        for batch in generator:
            if feature_vectors is None:
                feature_vectors = self.model.predict(batch)
            else:
                feature_vectors = np.concatenate((feature_vectors, self.model.predict(batch)))

        self.fitness = self._fitness_func(feature_vectors)

    def set_mutation_prob(self, mutation_prob):
        self._mutation_prob = mutation_prob

    def reset(self):
        self.model = self._model_generator()
        self.weights = np.random.normal(*self._distribution, self._size)
        self._update_model_weights()
        self.fitness = None

    def get_model(self):
        self._update_model_weights()
        return self.model

    def _update_model_weights(self):
        reshaped_weights = []
        start_idx = 0

        for idx in range(len(self._layer_shapes)):
            shape = self._layer_shapes[idx]
            end_idx = start_idx + np.prod(shape)
            reshaped_weights.append(
                self.weights[start_idx:end_idx].reshape(shape)
            )
            start_idx = end_idx

        self.model.set_weights(reshaped_weights)


class Population(object):
    def __init__(self, size, base_individual, mate_prob):
        self._size = size
        self._mate_prob = mate_prob
        self._population = [base_individual for _ in range(size)]

        for ind in self._population:
            ind.reset()

    def tournament(self, size):
        new_population = []
        for _ in range(self._size):
            new_population.append(max(
                np.random.choice(self._population, size)
            ))
        self._population = new_population

    def evolve(self, method, generator):
        if method == 'and':
            self._random_mating()
            self._random_mutation()
        elif method == 'or':
            if np.random.uniform() < 0.5:
                self._random_mutation()
            else:
                self._random_mating()
        else:
            raise ValueError('Method must be "and" or "or".')

        for individual in self._population:
            if individual.fitness is None:
                individual.evaluate_fitness(generator)

    @property
    def leader(self):
        return max(self._population)

    def _generate_random_mates(self):
        indexes = np.arange(len(self._population))
        for ind in indexes:
            mate = np.random.choice(np.delete(indexes, ind))
            yield self._population[ind], self._population[mate]

    def _random_mating(self):
        for individual, mate in self._generate_random_mates():
            if np.random.uniform() < self._mate_prob:
                individual * mate

    def _random_mutation(self):
        for individual in self._population:
            ~individual


def fitness_function(array):
    normalized = array / np.linalg.norm(array, axis=1).reshape(-1, 1)
    pairwise_euclidean_distances = distance.pdist(normalized, 'sqeuclidean')
    return np.min(pairwise_euclidean_distances) + np.mean(pairwise_euclidean_distances)


def thunderhill_generator(paths, batch_size=64):
    n_obs = paths.shape[0]
    batch_starts = np.arange(0, n_obs, batch_size)

    for batch in batch_starts:
        next_idx = batch + batch_size
        batch_x = paths[batch:min(next_idx, n_obs), ...]

        # Load the images from their paths
        loaded_ims = []
        for im_path in batch_x:
            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            loaded_ims.append(im)
        yield np.array(loaded_ims)


def train(generator,
          population,
          tournament_size,
          mate_and_or_mutate,
          max_generations,
          patience,
          min_delta=1e-5):
    fitness_history = []
    prev_fitness = 0
    generations_without_improvement = 0
    leader = None

    generations = trange(max_generations)
    for gen in generations:
        generations.set_description('GEN %i' % gen)

        population.evolve(method=mate_and_or_mutate, generator=generator)
        population.tournament(size=tournament_size)

        leader = population.leader
        fitness_history.append(leader.fitness)

        fitness_delta = fitness_history[-1] - prev_fitness
        if fitness_delta <= min_delta:
            generations_without_improvement += 1
            if generations_without_improvement == patience:
                break

        generations.set_postfix(fitness=fitness_history[-1], last_improvement=generations_without_improvement)
        prev_fitness = fitness_history[-1]
    return leader.get_model(), fitness_history


if __name__ == '__main__':
    np.random.seed(1234)

    individual = Individual(
        model_generator=models.evolutionary_feature_extractor,
        mu=0,
        sigma=1.5,
        mutation_prob=0.05,
        fitness_func=fitness_function
      )
    population = Population(
        size=100,
        base_individual=individual,
        mate_prob=0.8
      )

    # Load image paths
    path = os.getcwd() + '/'
    data = utils.load_polysync_paths()

    # Evolve the model
    best_model, history = train(
        generator=thunderhill_generator(data['center'], batch_size=64),
        population=population,
        tournament_size=10,
        mate_and_or_mutate='and',
        max_generations=100,
        patience=5,
        min_delta=1e-5
      )

    plt.plot(history)
    plt.title('Fitness History')
    best_model.save('feature_extractor.h5')
