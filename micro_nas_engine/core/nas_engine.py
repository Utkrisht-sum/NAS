import os
import re
from typing import Dict, Any, List

from core.search_space import SearchSpace
from core.evolutionary_search import EvolutionarySearch
from datasets.dataset_analyzer import analyze_dataset
from datasets.dataset_loader import load_dataset
from metrics.pareto import get_pareto_frontier
from knowledge.memory_manager import MemoryManager
from training.evaluator import evaluate_architecture
from utils.logger import logger
from utils.config import Config

class NASEngine:
    def __init__(self, dataset_source: str, prompt: str, pop_size: int = Config.DEFAULT_POPULATION_SIZE,
                 generations: int = Config.DEFAULT_GENERATIONS, compute_budget: int = Config.DEFAULT_COMPUTE_BUDGET,
                 progress_callback=None):
        self.dataset_source = dataset_source
        self.prompt = prompt
        self.pop_size = pop_size
        self.generations = generations
        self.compute_budget = compute_budget
        self.progress_callback = progress_callback

        # Internal State
        self.memory_manager = MemoryManager()
        self.alpha, self.beta = self._parse_prompt(prompt)
        self.best_architectures = [] # Pareto frontier across all generations
        self.all_evaluated = [] # All architectures evaluated
        self.dataset_meta = None
        self.train_loader = None
        self.val_loader = None

    def _parse_prompt(self, prompt: str) -> tuple[float, float]:
        """
        Parses the user prompt to infer fitness weights using a keyword heuristic.
        Returns: (alpha, beta) where alpha=accuracy_weight, beta=compute_penalty
        """
        prompt_lower = prompt.lower()
        alpha = Config.DEFAULT_ALPHA
        beta = Config.DEFAULT_BETA

        # Increase penalty for cost
        if any(word in prompt_lower for word in ["efficient", "fast", "lightweight", "small", "quick"]):
            beta *= 2.0
            logger.info("Prompt heuristic: Detected efficiency requirement. Increased compute penalty (beta).")

        # Increase reward for accuracy
        if any(word in prompt_lower for word in ["accurate", "best", "precise", "exact", "perfect"]):
            alpha *= 1.5
            logger.info("Prompt heuristic: Detected accuracy requirement. Increased accuracy reward (alpha).")

        return alpha, beta

    def run(self) -> Dict[str, Any]:
        """Executes the complete NAS pipeline."""
        logger.info(f"Starting NAS Engine on {self.dataset_source}")

        # 1. Analyze Dataset
        self.dataset_meta = analyze_dataset(self.dataset_source)
        logger.info(f"Dataset Analysis: {self.dataset_meta}")

        # Map dataset type to search space type
        if self.dataset_meta["type"] == "tabular":
            search_space_type = "tabular"
        elif self.dataset_meta["type"] == "image":
            search_space_type = "image"
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_meta['type']}")

        # 2. Load Data
        self.train_loader, self.val_loader, loaded_meta = load_dataset(self.dataset_source)
        # Update metadata with actual loaded details
        self.dataset_meta.update(loaded_meta)

        # 3. Initialize Search Space and Evolutionary Algorithm
        search_space = SearchSpace(search_space_type)
        evo_search = EvolutionarySearch(search_space, self.alpha, self.beta, self.pop_size)

        # 4. Load Past Knowledge (Seeding)
        past_archs = self.memory_manager.get_past_architectures(search_space_type, limit=self.pop_size // 2)
        population = evo_search.initialize_population(past_archs)

        # 5. Evolution Loop
        for gen in range(self.generations):
            logger.info(f"--- Generation {gen + 1}/{self.generations} ---")

            gen_results = []

            # Evaluate all individuals in population
            for i, arch in enumerate(population):
                if self.progress_callback:
                    # Report overall generation progress before evaluating individual
                    # e.g., callback(current_gen, total_gens, current_ind, total_inds, ...)
                    self.progress_callback(gen + 1, self.generations, i + 1, self.pop_size, "Evaluating...", 0, 0)

                # Evaluate
                res = evaluate_architecture(
                    arch,
                    input_dim=self.dataset_meta["input_dim"],
                    num_classes=self.dataset_meta["num_classes"],
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    task_type=self.dataset_meta.get("task_type", "classification"),
                    epochs=2 # Short training for proxy evaluation
                )

                # Check budget constraint (Hard filter)
                if res["compute_cost"] > self.compute_budget:
                    logger.warning(f"Architecture exceeded compute budget: {res['compute_cost']} > {self.compute_budget}. Assigning poor fitness.")
                    res["accuracy"] = -1.0 # Guarantee poor rank

                gen_results.append(res)
                self.all_evaluated.append(res)

            # Rank and Select
            ranked = evo_search.rank_population(gen_results)
            logger.info(f"Generation {gen+1} Best Accuracy: {ranked[0]['accuracy']:.4f}")

            # Create next generation if not the last
            if gen < self.generations - 1:
                population = evo_search.step_generation(ranked)

        # 6. Final Pareto Selection
        # Filter out failed evaluations (-1.0 accuracy)
        valid_evaluations = [r for r in self.all_evaluated if r["accuracy"] > -1.0]
        self.best_architectures = get_pareto_frontier(valid_evaluations)

        # 7. Final Output & Memory Update
        if not self.best_architectures:
            logger.error("No valid architectures found within budget.")
            return None

        # The "absolute best" is the one on the Pareto frontier with the highest fitness score
        # Recalculate fitness to ensure consistency
        from metrics.fitness import calculate_fitness
        best_overall = max(self.best_architectures, key=lambda x: calculate_fitness(x["accuracy"], x["compute_cost"], self.alpha, self.beta))

        logger.info(f"NAS Complete. Best Architecture: {best_overall['architecture']}")
        logger.info(f"Accuracy: {best_overall['accuracy']:.4f}, Params: {best_overall['compute_cost']}, FLOPs: {best_overall['flops']}")

        # Save to memory
        self.memory_manager.add_entry(
            self.dataset_meta,
            best_overall["architecture"],
            best_overall["accuracy"],
            best_overall["compute_cost"]
        )

        return {
            "best_overall": best_overall,
            "pareto_frontier": self.best_architectures,
            "all_evaluated": valid_evaluations
        }
