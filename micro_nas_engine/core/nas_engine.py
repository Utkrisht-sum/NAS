import random
import copy
from typing import List, Dict, Any
import numpy as np

from micro_nas_engine.models.search_space import sample_architecture
from micro_nas_engine.models.builder import build_model
from micro_nas_engine.metrics.zero_cost import filter_architectures
from micro_nas_engine.metrics.profiler import estimate_compute_cost, estimate_flops
from micro_nas_engine.training.trainer import ProxyTrainer
from micro_nas_engine.knowledge.memory import ArchitectureMemory

class NASEngine:
    def __init__(self, task_type, input_shape, num_classes, dataloader_train, dataloader_val,
                 device='cpu', accelerator=None, config=None):
        self.task_type = task_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.device = device
        self.accelerator = accelerator

        self.config = config or {}
        self.pop_size = self.config.get('pop_size', 10)
        self.generations = self.config.get('generations', 5)
        self.alpha = self.config.get('alpha', 1.0)
        self.beta = self.config.get('beta', 1e-6)

        self.memory = ArchitectureMemory()
        self.proxy_trainer = ProxyTrainer(accelerator=self.accelerator, device=self.device)
        self.history = []

    def fitness_function(self, accuracy, compute_cost):
        return self.alpha * accuracy - self.beta * compute_cost

    def _mutate(self, arch_config):
        mutated = copy.deepcopy(arch_config)
        if self.task_type == "Tabular":
            # Mutation for MLP
            if random.random() < 0.5:
                # Add/remove layer
                if random.random() < 0.5 and mutated["num_layers"] < 5:
                    mutated["num_layers"] += 1
                    mutated["hidden_sizes"].append(random.choice([16, 32, 64, 128]))
                elif mutated["num_layers"] > 1:
                    mutated["num_layers"] -= 1
                    mutated["hidden_sizes"].pop()
            else:
                # Change activation
                mutated["activation"] = random.choice(['ReLU', 'Tanh', 'SiLU'])
        elif self.task_type == "Image":
            # Mutation for CNN
            cells = mutated.get("cells", [])
            if cells:
                cell_idx = random.randint(0, len(cells)-1)
                node_idx = random.randint(0, len(cells[cell_idx])-1)

                # Resample this node's operations
                from micro_nas_engine.models.search_space import OPS_CNN
                new_op1 = random.choice(list(OPS_CNN.keys()))
                new_op2 = random.choice(list(OPS_CNN.keys()))

                # Maintain same input connections for simplicity in mutation
                old_node = cells[cell_idx][node_idx]
                cells[cell_idx][node_idx] = ((new_op1, old_node[0][1] if isinstance(old_node[0], tuple) else 0),
                                             (new_op2, old_node[1][1] if len(old_node) > 1 and isinstance(old_node[1], tuple) else 0))
        return mutated

    def bayesian_optimization_step(self, population):
        """
        Stage 5: Bayesian Optimization Approximation (TPE-like).
        Instead of fully training a Gaussian Process which is complex without a library,
        we use the history to bias our random sampling towards areas of the search space
        that performed well (Tree-structured Parzen Estimator approximation).
        """
        if len(self.history) < self.pop_size:
            return population # Not enough history

        # Sort history by fitness
        sorted_history = sorted(self.history, key=lambda x: x['fitness'], reverse=True)
        good_configs = [x['config'] for x in sorted_history[:len(sorted_history)//3]]

        # Create new candidates by mutating the top configs from history
        bo_candidates = []
        for _ in range(self.pop_size):
            parent = random.choice(good_configs)
            bo_candidates.append(self._mutate(parent))

        return bo_candidates

    def darts_refinement(self, best_config):
        """
        Stage 6: Optional DARTS refinement for top candidate architecture.
        Since true DARTS requires a differentiable supernet, we approximate this local
        refinement by running a small grid search on the hyperparameters of the best architecture.
        """
        refined = copy.deepcopy(best_config)

        if self.task_type == "Tabular":
            # Just test tweaking the activation or adding a slight size bump
            refined["hidden_sizes"] = [int(s * 1.1) for s in refined["hidden_sizes"]]

        elif self.task_type == "Image":
             # Change max pooling to avg pooling in best architecture if it exists
             cells = refined.get("cells", [])
             for i in range(len(cells)):
                 for j in range(len(cells[i])):
                     node = cells[i][j]
                     if isinstance(node[0], tuple) and node[0][0] == 'max_pool_3x3':
                         cells[i][j] = (('avg_pool_3x3', node[0][1]), node[1])
                     if isinstance(node[1], tuple) and node[1][0] == 'max_pool_3x3':
                         cells[i][j] = (node[0], ('avg_pool_3x3', node[1][1]))

        return refined

    def search(self, progress_callback=None):
        """
        Executes the staged NAS pipeline.
        """
        # Stage 1: Architecture Sampling
        if progress_callback: progress_callback("Stage 1: Architecture Sampling")
        initial_pool_size = self.pop_size * 5
        pool_configs = [sample_architecture(self.task_type) for _ in range(initial_pool_size)]
        pool_models = [build_model(self.task_type, self.input_shape, self.num_classes, cfg) for cfg in pool_configs]

        # Stage 2: Zero-cost filtering
        if progress_callback: progress_callback("Stage 2: Zero-cost filtering")
        models_configs = list(zip(pool_models, pool_configs))
        filtered_configs = filter_architectures(models_configs, self.dataloader_train, self.device, keep_ratio=0.2)

        # Take the top ones to form initial population
        population = [cfg for _, cfg in filtered_configs[:self.pop_size]]

        # Evolutionary Search
        for gen in range(self.generations):
            if progress_callback: progress_callback(f"Stage 4: Evolutionary Search - Generation {gen+1}/{self.generations}")

            evaluated_pop = []
            for cfg in population:
                model = build_model(self.task_type, self.input_shape, self.num_classes, cfg)
                cost = estimate_compute_cost(model, self.task_type, cfg)

                # Stage 3: Proxy training
                acc = self.proxy_trainer.train_model(model, self.dataloader_train, self.dataloader_val, epochs=1, subset_ratio=0.2)

                fitness = self.fitness_function(acc, cost)
                evaluated_pop.append({"config": cfg, "accuracy": acc, "cost": cost, "fitness": fitness})
                self.history.append(evaluated_pop[-1])

            # Sort by fitness
            evaluated_pop.sort(key=lambda x: x["fitness"], reverse=True)

            # Stage 5: Bayesian optimization step periodically
            if gen > 0 and gen % 2 == 0:
                if progress_callback: progress_callback(f"Stage 5: Bayesian Optimization Step")
                bo_pop = self.bayesian_optimization_step(population)
                population = bo_pop
            else:
                # Selection & Evolutionary crossover
                top_half = [item["config"] for item in evaluated_pop[:self.pop_size // 2]]
                new_population = list(top_half)
                while len(new_population) < self.pop_size:
                    parent = random.choice(top_half)
                    child = self._mutate(parent)
                    new_population.append(child)
                population = new_population

        # Final Evaluation
        final_evaluated = []
        for cfg in population:
            model = build_model(self.task_type, self.input_shape, self.num_classes, cfg)
            cost = estimate_compute_cost(model, self.task_type, cfg)
            acc = self.proxy_trainer.train_model(model, self.dataloader_train, self.dataloader_val, epochs=2, subset_ratio=0.5)
            fitness = self.fitness_function(acc, cost)
            final_evaluated.append({"config": cfg, "accuracy": acc, "cost": cost, "fitness": fitness})
            self.history.append(final_evaluated[-1])
            self.memory.add_entry({"type": self.task_type}, cfg, acc, cost)

        final_evaluated.sort(key=lambda x: x["fitness"], reverse=True)

        # Stage 6: DARTS Refinement on top candidate
        if progress_callback: progress_callback(f"Stage 6: DARTS Refinement")
        best_cfg = final_evaluated[0]["config"]
        refined_cfg = self.darts_refinement(best_cfg)

        # Evaluate refined config
        model = build_model(self.task_type, self.input_shape, self.num_classes, refined_cfg)
        cost = estimate_compute_cost(model, self.task_type, refined_cfg)
        acc = self.proxy_trainer.train_model(model, self.dataloader_train, self.dataloader_val, epochs=2, subset_ratio=0.5)
        refined_fitness = self.fitness_function(acc, cost)

        self.history.append({"config": refined_cfg, "accuracy": acc, "cost": cost, "fitness": refined_fitness})

        if refined_fitness > final_evaluated[0]["fitness"]:
            if progress_callback: progress_callback(f"DARTS refinement improved architecture.")
            final_evaluated[0] = {"config": refined_cfg, "accuracy": acc, "cost": cost, "fitness": refined_fitness}

        # Sort again just in case
        final_evaluated.sort(key=lambda x: x["fitness"], reverse=True)
        return final_evaluated

    def get_pareto_frontier(self):
        """
        Returns a list of architectures that form the Pareto frontier
        (maximizing accuracy, minimizing cost).
        """
        frontier = []
        sorted_history = sorted(self.history, key=lambda x: x['accuracy'], reverse=True)

        min_cost = float('inf')
        for item in sorted_history:
            if item['cost'] < min_cost:
                frontier.append(item)
                min_cost = item['cost']

        return frontier
