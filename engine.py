import random
import copy
from dna.dna import EvoDNA
from engine.trainer import train_and_evaluate

class EvolutionEngine:
    def __init__(self, population_size=10, mutation_rate=0.4, device="cpu"):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.device = device
        self.population = [EvoDNA() for _ in range(population_size)]
        self.fitness_scores = [0.0 for _ in range(population_size)]

    def evaluate_population(self):
        self.fitness_scores = [
            train_and_evaluate(dna, device=self.device) for dna in self.population
        ]

    def select_parents(self):
        # Tournament selection
        idxs = random.sample(range(len(self.population)), 4)
        idxs.sort(key=lambda i: self.fitness_scores[i], reverse=True)
        return self.population[idxs[0]], self.population[idxs[1]]

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        if random.random() < 0.5:
            child.optimizer = copy.deepcopy(parent2.optimizer)
        if random.random() < 0.5:
            child.loss_function = parent2.loss_function
        if random.random() < 0.5:
            child.lr_schedule = parent2.lr_schedule
        return child

    def evolve(self):
        self.evaluate_population()

        # Elitism: keep top 2
        sorted_indices = sorted(range(len(self.population)), key=lambda i: self.fitness_scores[i], reverse=True)
        new_population = [copy.deepcopy(self.population[i]) for i in sorted_indices[:2]]

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            if random.random() < self.mutation_rate:
                child.mutate()
            new_population.append(child)

        self.population = new_population
