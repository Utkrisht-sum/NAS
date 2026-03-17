import random
import copy
from typing import List, Dict, Any
import numpy as np
import torch

from micro_nas_engine.models.search_space import sample_architecture, validate_cnn_spatial_dimensions, repair_architecture
from micro_nas_engine.models.builder import build_model
from micro_nas_engine.metrics.zero_cost import filter_architectures
from micro_nas_engine.metrics.profiler import estimate_compute_cost, estimate_flops, is_architecture_memory_safe, estimate_latency
from micro_nas_engine.training.trainer import ProxyTrainer
from micro_nas_engine.knowledge.memory import ArchitectureMemory

class FailureRegistry:
    def __init__(self):
        self.failures = []

    def add_failure(self, dna, reason):
        self.failures.append({"dna": dna, "reason": reason})

    def is_known_failure(self, dna):
        return any(f["dna"] == dna for f in self.failures)

    def get_report(self):
        return self.failures

class HybridPredictor:
    def __init__(self):
        self.dna_to_fitness = {}

    def add_data(self, dna, fitness):
        self.dna_to_fitness[dna] = fitness

    def predict(self, dna):
        # Very simple nearest-neighbor baseline for DNA string matching
        if not self.dna_to_fitness:
            return 0.0

        # Find closest match based on string overlap
        best_match_dna = max(self.dna_to_fitness.keys(), key=lambda k: sum(a==b for a,b in zip(dna, k)))
        return self.dna_to_fitness[best_match_dna]

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
        # Default to Fast NAS mode parameters
        self.pop_size = self.config.get('pop_size', 5)
        self.generations = self.config.get('generations', 2)

        # Hardware-aware fitness weights
        self.alpha = self.config.get('alpha', 1.0)      # accuracy
        self.beta = self.config.get('beta', 1e-6)       # params/cost
        self.gamma = self.config.get('gamma', 1e-4)     # latency
        self.delta = self.config.get('delta', 1e-4)     # memory

        self.memory = ArchitectureMemory()
        self.proxy_trainer = ProxyTrainer(accelerator=self.accelerator, device=self.device)
        self.history = []
        self.failures = FailureRegistry()
        self.predictor = HybridPredictor()

        self.evaluated_dnas = set()

    def fitness_function(self, accuracy, compute_cost, latency, memory_mb):
        return (self.alpha * accuracy) - (self.beta * compute_cost) - (self.gamma * latency) - (self.delta * memory_mb)

    def _mutate(self, arch_config):
        mutated = copy.deepcopy(arch_config)
        # Apply slight mutations
        if self.task_type == "Tabular":
            if random.random() < 0.5:
                if random.random() < 0.5 and mutated["num_layers"] < 5:
                    mutated["num_layers"] += 1
                    mutated["hidden_sizes"].append(random.choice([16, 32, 64, 128]))
                elif mutated["num_layers"] > 1:
                    mutated["num_layers"] -= 1
                    mutated["hidden_sizes"].pop()
            else:
                mutated["activation"] = random.choice(['ReLU', 'Tanh', 'SiLU'])
            mutated["dna"] = f"L{mutated['num_layers']}S{sum(mutated['hidden_sizes'])}A{mutated['activation']}"
        elif self.task_type == "Image":
            cells = mutated.get("cells", [])
            if cells:
                cell_idx = random.randint(0, len(cells)-1)
                node_idx = random.randint(0, len(cells[cell_idx])-1)
                from micro_nas_engine.models.search_space import OPS_CNN
                new_op1 = random.choice(list(OPS_CNN.keys()))
                new_op2 = random.choice(list(OPS_CNN.keys()))
                old_node = cells[cell_idx][node_idx]
                cells[cell_idx][node_idx] = ((new_op1, old_node[0][1] if isinstance(old_node[0], tuple) else 0),
                                             (new_op2, old_node[1][1] if len(old_node) > 1 and isinstance(old_node[1], tuple) else 0))
            mutated["dna"] = f"C{len(mutated.get('cells', []))}N{len(mutated.get('cells', [[]])[0])//2 if mutated.get('cells') else 0}"
        return mutated

    def bayesian_optimization_step(self, population):
        if len(self.history) < self.pop_size:
            return population
        sorted_history = sorted(self.history, key=lambda x: x['fitness'], reverse=True)
        good_configs = [x['config'] for x in sorted_history[:max(1, len(sorted_history)//3)]]
        bo_candidates = []
        for _ in range(self.pop_size):
            parent = random.choice(good_configs)
            bo_candidates.append(self._mutate(parent))
        return bo_candidates

    def darts_refinement(self, best_config):
        refined = copy.deepcopy(best_config)
        if self.task_type == "Tabular":
            refined["hidden_sizes"] = [int(s * 1.1) for s in refined["hidden_sizes"]]
        elif self.task_type == "Image":
             cells = refined.get("cells", [])
             for i in range(len(cells)):
                 for j in range(len(cells[i])):
                     node = cells[i][j]
                     if isinstance(node[0], tuple) and node[0][0] == 'max_pool_3x3':
                         cells[i][j] = (('avg_pool_3x3', node[0][1]), node[1])
                     if isinstance(node[1], tuple) and node[1][0] == 'max_pool_3x3':
                         cells[i][j] = (node[0], ('avg_pool_3x3', node[1][1]))
        return refined

    def validate_and_repair(self, config):
        """ Checks safety constraints. If invalid, attempts repair. """
        dna = config.get("dna", "")

        if self.failures.is_known_failure(dna):
            return False, config, "Known Failure"

        # Check spatial dimension safety
        is_valid, _ = validate_cnn_spatial_dimensions(config, self.input_shape)
        if not is_valid:
            repaired = repair_architecture(config, self.task_type)
            self.failures.add_failure(dna, "Invalid Spatial Dimensions")
            return True, repaired, "Repaired Spatial"

        # Check Memory safety
        try:
            model = build_model(self.task_type, self.input_shape, self.num_classes, config)
            is_mem_safe, est_mem = is_architecture_memory_safe(model, self.input_shape, batch_size=32)
            if not is_mem_safe:
                repaired = repair_architecture(config, self.task_type)
                self.failures.add_failure(dna, f"OOM Limit Exceeded ({est_mem:.1f} MB)")
                return True, repaired, "Repaired Memory"
        except Exception as e:
            self.failures.add_failure(dna, f"Build Error: {str(e)}")
            return False, config, "Build Error"

        return True, config, "Valid"

    def search(self, progress_callback=None):
        if progress_callback: progress_callback("Stage 1: Architecture Sampling (Memory Guarded)")

        initial_pool_size = self.pop_size * 5
        pool_configs = []
        pool_models = []

        # Sampling with Validation
        while len(pool_configs) < initial_pool_size:
            cfg = sample_architecture(self.task_type)
            dna = cfg.get("dna", "")

            # Architecture Similarity Pruning
            if dna in self.evaluated_dnas:
                continue

            is_valid, final_cfg, status = self.validate_and_repair(cfg)
            if is_valid:
                model = build_model(self.task_type, self.input_shape, self.num_classes, final_cfg)
                pool_configs.append(final_cfg)
                pool_models.append(model)
                self.evaluated_dnas.add(final_cfg.get("dna", ""))

        if progress_callback: progress_callback("Stage 2: Zero-cost filtering")
        models_configs = list(zip(pool_models, pool_configs))
        filtered_configs = filter_architectures(models_configs, self.dataloader_train, self.device, keep_ratio=0.2)
        population = [cfg for _, cfg in filtered_configs[:self.pop_size]]

        # Evolutionary Search
        for gen in range(self.generations):
            if progress_callback: progress_callback(f"Stage 4: Evolutionary Search - Gen {gen+1}/{self.generations}")

            evaluated_pop = []
            for cfg in population:
                model = build_model(self.task_type, self.input_shape, self.num_classes, cfg)
                cost = estimate_compute_cost(model, self.task_type, cfg)
                _, memory_mb = is_architecture_memory_safe(model, self.input_shape)
                latency = estimate_latency(model, self.input_shape)

                # Hybrid prediction check
                pred_fitness = self.predictor.predict(cfg.get("dna", ""))

                # Fast proxy training (2 epochs)
                try:
                    acc = self.proxy_trainer.train_model(model, self.dataloader_train, self.dataloader_val, epochs=2, subset_ratio=0.2)
                except Exception as e:
                    self.failures.add_failure(cfg.get("dna", ""), f"Training crashed: {str(e)}")
                    continue # Skip to next candidate on crash

                fitness = self.fitness_function(acc, cost, latency, memory_mb)
                self.predictor.add_data(cfg.get("dna", ""), fitness)

                evaluated_pop.append({"config": cfg, "accuracy": acc, "cost": cost, "memory": memory_mb, "fitness": fitness})
                self.history.append(evaluated_pop[-1])

            evaluated_pop.sort(key=lambda x: x["fitness"], reverse=True)

            if gen > 0 and gen % 2 == 0:
                if progress_callback: progress_callback(f"Stage 5: Bayesian Optimization Step")
                bo_pop = self.bayesian_optimization_step([item["config"] for item in evaluated_pop])

                # Validate BO pop
                population = []
                for cfg in bo_pop:
                    is_valid, final_cfg, _ = self.validate_and_repair(cfg)
                    if is_valid and final_cfg.get("dna") not in self.evaluated_dnas:
                        population.append(final_cfg)
                        self.evaluated_dnas.add(final_cfg.get("dna", ""))
            else:
                top_half = [item["config"] for item in evaluated_pop[:max(1, len(evaluated_pop) // 2)]]
                new_population = list(top_half)
                while len(new_population) < self.pop_size:
                    parent = random.choice(top_half)
                    child = self._mutate(parent)
                    is_valid, final_child, _ = self.validate_and_repair(child)
                    if is_valid and final_child.get("dna") not in self.evaluated_dnas:
                        new_population.append(final_child)
                        self.evaluated_dnas.add(final_child.get("dna", ""))
                population = new_population

        # Final Evaluation
        final_evaluated = []
        for cfg in population:
            model = build_model(self.task_type, self.input_shape, self.num_classes, cfg)
            cost = estimate_compute_cost(model, self.task_type, cfg)
            _, memory_mb = is_architecture_memory_safe(model, self.input_shape)
            latency = estimate_latency(model, self.input_shape)
            try:
                acc = self.proxy_trainer.train_model(model, self.dataloader_train, self.dataloader_val, epochs=2, subset_ratio=0.5)
                fitness = self.fitness_function(acc, cost, latency, memory_mb)
                final_evaluated.append({"config": cfg, "accuracy": acc, "cost": cost, "memory": memory_mb, "fitness": fitness})
                self.history.append(final_evaluated[-1])
                self.memory.add_entry({"type": self.task_type}, cfg, acc, cost)
            except Exception as e:
                pass

        if not final_evaluated:
            # Fallback to history if everything failed
            final_evaluated = sorted(self.history, key=lambda x: x['fitness'], reverse=True)

        final_evaluated.sort(key=lambda x: x["fitness"], reverse=True)

        # Stage 6: DARTS Refinement on top candidate
        if final_evaluated:
            if progress_callback: progress_callback(f"Stage 6: DARTS Refinement")
            best_cfg = final_evaluated[0]["config"]
            refined_cfg = self.darts_refinement(best_cfg)

            is_valid, final_refined_cfg, _ = self.validate_and_repair(refined_cfg)
            if is_valid:
                model = build_model(self.task_type, self.input_shape, self.num_classes, final_refined_cfg)
                cost = estimate_compute_cost(model, self.task_type, final_refined_cfg)
                _, memory_mb = is_architecture_memory_safe(model, self.input_shape)
                latency = estimate_latency(model, self.input_shape)
                try:
                    acc = self.proxy_trainer.train_model(model, self.dataloader_train, self.dataloader_val, epochs=2, subset_ratio=0.5)
                    refined_fitness = self.fitness_function(acc, cost, latency, memory_mb)
                    self.history.append({"config": final_refined_cfg, "accuracy": acc, "cost": cost, "memory": memory_mb, "fitness": refined_fitness})

                    if refined_fitness > final_evaluated[0]["fitness"]:
                        if progress_callback: progress_callback(f"DARTS refinement improved architecture.")
                        final_evaluated[0] = {"config": final_refined_cfg, "accuracy": acc, "cost": cost, "memory": memory_mb, "fitness": refined_fitness}
                except:
                    pass

        final_evaluated.sort(key=lambda x: x["fitness"], reverse=True)
        return final_evaluated

    def get_pareto_frontier(self):
        frontier = []
        sorted_history = sorted(self.history, key=lambda x: x['accuracy'], reverse=True)
        min_cost = float('inf')
        for item in sorted_history:
            if item['cost'] < min_cost:
                frontier.append(item)
                min_cost = item['cost']
        return frontier
