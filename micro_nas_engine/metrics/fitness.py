def calculate_fitness(accuracy: float, compute_cost: float, alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Computes fitness function F(A) for architecture A.
    F(A) = α * Accuracy - β * ComputeCost

    Since ComputeCost can be large (millions of parameters),
    we normalize it using a scaled penalty to make it comparable to accuracy (0-1 scale).
    """

    # We penalize based on a log-scale or a scaled factor of the compute cost
    import math

    # Soft log scaling for parameter count to prevent dominant penalty
    normalized_cost = math.log10(max(1, compute_cost)) / 10.0 # scale cost down

    fitness = (alpha * accuracy) - (beta * normalized_cost)
    return fitness
