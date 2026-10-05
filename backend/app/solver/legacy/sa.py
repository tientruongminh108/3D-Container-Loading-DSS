import random
import math
import copy
from typing import List, Callable, Optional, Tuple, TYPE_CHECKING
from app.config import get_settings
from app.solver.parsing import Box
from app.solver.geometry import Dimensions
from app.solver.fitness import calculate_fitness, FitnessResult

if TYPE_CHECKING:
    from app.solver.ga import Individual


def run_simulated_annealing(
    container_dims: Dimensions,
    boxes_sorted: List[Box],
    best_individual: "Individual",
    best_fitness: float,
    is_lcl: bool = False,
    max_weight: float = 0.0,
    auto_tune: bool = False,
) -> Tuple["Individual", float]:
    """
    Simulated Annealing as local refinement operator on GA's best individual.
    Per guide Section 5.3.4: perturbs single posture, evaluates via decode/fitness.
    """
    settings = get_settings()

    current = copy.deepcopy(best_individual)
    current_fitness = best_fitness
    sa_best = copy.deepcopy(best_individual)
    sa_best_fitness = best_fitness

    # Auto-tune initial temperature from fitness delta standard deviation
    if auto_tune:
        deltas = []
        for _ in range(50):
            neighbor = generate_neighbor(current, boxes_sorted)
            evaluate_individual(neighbor, boxes_sorted, container_dims, max_weight, is_lcl)
            delta = abs(neighbor.fitness_result.fitness - current_fitness)
            deltas.append(delta)
        if deltas:
            sigma = math.sqrt(sum((d - sum(deltas)/len(deltas))**2 for d in deltas) / len(deltas))
            temperature = 2 * sigma
        else:
            temperature = settings.SA_INITIAL_TEMP
    else:
        temperature = settings.SA_INITIAL_TEMP

    min_temperature = settings.SA_MIN_TEMP
    cooling_rate = settings.SA_COOLING_RATE

    steps = 0
    max_steps = 15
    while temperature > min_temperature and steps < max_steps:
        steps += 1
        neighbor = generate_neighbor(current, boxes_sorted)
        from app.solver.ga import evaluate_individual
        evaluate_individual(neighbor, boxes_sorted, container_dims, max_weight, is_lcl)

        delta = neighbor.fitness_result.fitness - current_fitness

        if delta > 0 or random.random() < math.exp(delta / temperature):
            current = neighbor
            current_fitness = neighbor.fitness_result.fitness
            if current_fitness > sa_best_fitness:
                sa_best = copy.deepcopy(current)
                sa_best_fitness = current_fitness

        temperature *= cooling_rate

    return sa_best, sa_best_fitness


def generate_neighbor(individual: "Individual", units: List[Box]) -> "Individual":
    """Multi-gene SA perturbation.

    50% probability: single posture flip.
    30% probability: flip 2-3 random posture genes.
    20% probability: swap postures of two randomly chosen items.
    """
    import random as _random
    neighbor = individual.clone()

    if not neighbor.chromosome:
        return neighbor

    n = len(neighbor.chromosome)
    roll = _random.random()

    if roll < 0.50:
        # Single-gene flip
        idx = _random.randrange(n)
        if idx < len(units) and units[idx].permitted_postures:
            neighbor.chromosome[idx] = _random.randrange(len(units[idx].permitted_postures))

    elif roll < 0.80:
        # Multi-gene: flip 2 or 3 random genes
        num_flips = _random.choice([2, 3])
        indices = _random.sample(range(n), min(num_flips, n))
        for idx in indices:
            if idx < len(units) and units[idx].permitted_postures:
                neighbor.chromosome[idx] = _random.randrange(len(units[idx].permitted_postures))

    else:
        # Swap: exchange posture genes of two randomly chosen items
        if n >= 2:
            i, j = _random.sample(range(n), 2)
            neighbor.chromosome[i], neighbor.chromosome[j] = (
                neighbor.chromosome[j],
                neighbor.chromosome[i],
            )
    
    # --- Order-gene move ---
    if neighbor.order_keys is None or len(neighbor.order_keys) != n:
        neighbor.order_keys = [float(i) for i in range(n)]

    if n >= 2:
        order_roll = _random.random()
        order = neighbor.placement_order(units)

        if order_roll < 0.4:
            p = _random.randrange(n - 1)
            i, j = order[p], order[p + 1]
            neighbor.order_keys[i], neighbor.order_keys[j] = (
                neighbor.order_keys[j],
                neighbor.order_keys[i],
            )
        
        elif order_roll < 0.75:
            p = _random.randrange(n)
            shift = _random.randint(-3, 3)
            target = max(0, min(n - 1, p + shift))
            
            if target != p:
                i = order[p]
                lo = neighbor.order_keys[order[target]]
                if target > 0:
                    hi_idx = order[target - 1] if target < p else order[min(target + 1, n - 1)]
                else:
                    hi_idx = order[min(target + 1, n - 1)]
                hi = neighbor.order_keys[hi_idx]
                neighbor.order_keys[i] = (lo + hi) / 2.0
        else:
            idx = _random.randrange(n)
            neighbor.order_keys[idx] += _random.uniform(-1.5, 1.5)
    
    return neighbor


def simulated_annealing(
    individual: "Individual",
    units: List[Box],
    container_dims: Dimensions,
    max_weight: float,
    is_lcl: bool = False,
    initial_temp: float = None,
    cooling_rate: float = None,
    iterations_per_temp: int = None,
    invoke_callback: Callable[[int, "Individual"], None] = None,
) -> "Individual":
    """Legacy SA interface."""
    settings = get_settings()

    temp = initial_temp or settings.SA_INITIAL_TEMP
    cooling = cooling_rate or settings.SA_COOLING_RATE
    iters = iterations_per_temp or 10

    current = copy.deepcopy(individual)
    best = copy.deepcopy(individual)

    iteration = 0

    while temp > 0.01:
        for _ in range(iters):
            neighbor = generate_neighbor(current, units)

            from app.solver.ga import evaluate_individual
            evaluate_individual(neighbor, units, container_dims, max_weight, is_lcl)

            delta = neighbor.fitness_result.fitness - current.fitness_result.fitness

            if delta > 0 or random.random() < math.exp(delta / temp):
                current = neighbor
                if current.fitness_result.fitness > best.fitness_result.fitness:
                    best = copy.deepcopy(current)

            iteration += 1
            if invoke_callback:
                invoke_callback(iteration, best)

        temp *= cooling

    return best
