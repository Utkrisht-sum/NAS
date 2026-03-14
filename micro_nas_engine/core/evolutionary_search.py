import random
from typing import List, Dict, Any, Tuple

from core.search_space import SearchSpace
from metrics.fitness import calculate_fitness
from utils.logger import logger

class EvolutionarySearch:
    def __init__(self, search_space: SearchSpace, alpha: float, beta: float, pop_size: int = 10):
        self.search_space = search_space
        self.alpha = alpha
        self.beta = beta
        self.pop_size = pop_size
        self.population: List[Dict[str, Any]] = []

    def initialize_population(self, seed_architectures: List[Dict[str, Any]] = None):
        """Initializes the starting population."""
        self.population = []

        # Add seed architectures if available (Knowledge Transfer)
        if seed_architectures:
            for arch in seed_architectures[:self.pop_size // 2]:
                # Deep copy to avoid mutating the original
                import copy
                self.population.append(copy.deepcopy(arch))

        # Fill the rest with random candidates
        while len(self.population) < self.pop_size:
            self.population.append(self.search_space.sample_random_architecture())

        logger.info(f"Initialized population of size {self.pop_size} ({len(seed_architectures) if seed_architectures else 0} seeded)")
        return self.population

    def rank_population(self, evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks the current population based on their fitness function score.
        evaluation_results: List of dicts containing 'architecture', 'accuracy', 'compute_cost'.
        """
        scored_population = []
        for res in evaluation_results:
            fitness = calculate_fitness(res["accuracy"], res["compute_cost"], self.alpha, self.beta)
            res["fitness"] = fitness
            scored_population.append(res)

        # Sort by fitness descending (higher is better)
        scored_population.sort(key=lambda x: x["fitness"], reverse=True)
        return scored_population

    def step_generation(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given the ranked results of the current generation, creates the next generation
        using elitism, mutation, and crossover.
        """
        new_population = []

        # Elitism: Keep the top 2 architectures unchanged
        top_k = min(2, len(ranked_results))
        for i in range(top_k):
            # Important: extract just the architecture dict
            new_population.append(ranked_results[i]["architecture"])

        # Selection Pool (top 50%)
        selection_pool = [res["architecture"] for res in ranked_results[:len(ranked_results)//2]]

        # Fill the rest
        while len(new_population) < self.pop_size:
            if random.random() < 0.7:
                # 70% chance to mutate a strong candidate
                parent = random.choice(selection_pool)
                child = self._mutate(parent)
                new_population.append(child)
            else:
                # 30% chance to generate entirely new random candidate (exploration)
                new_population.append(self.search_space.sample_random_architecture())

        self.population = new_population
        return self.population

    def _mutate(self, arch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutates a single architecture by changing one of its hyperparameters.
        """
        import copy
        mutated = copy.deepcopy(arch)
        mutation_target = random.choice(["activation", "dropout", "hidden_units", "filters", "kernel_sizes"])

        try:
            if mutation_target == "activation":
                mutated["activation"] = random.choice(self.search_space.activations)
            elif mutation_target == "dropout":
                mutated["dropout"] = round(random.uniform(self.search_space.dropout_min, self.search_space.dropout_max), 2)
            elif mutation_target == "hidden_units":
                # Change size of a random hidden layer
                idx = random.randint(0, len(mutated["hidden_units"]) - 1)
                mutated["hidden_units"][idx] = random.choice([16, 32, 64, 128, 256, 512])
            elif mutation_target == "filters" and mutated["type"] == "cnn":
                # Change filters of a random conv layer
                idx = random.randint(0, len(mutated["filters"]) - 1)
                mutated["filters"][idx] = random.choice([16, 32, 64, 128, 256])
            elif mutation_target == "kernel_sizes" and mutated["type"] == "cnn":
                idx = random.randint(0, len(mutated["kernel_sizes"]) - 1)
                mutated["kernel_sizes"][idx] = random.choice(self.search_space.cnn_kernel_choices)
        except Exception as e:
            # If a mutation fails (e.g. trying to mutate filters on MLP), ignore and return original
            pass

        return mutated
