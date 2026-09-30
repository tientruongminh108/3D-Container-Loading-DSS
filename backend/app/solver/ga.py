import random
import copy
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from app.config import get_settings
from app.solver.parsing import Box
from app.solver.block_generation import Block
from app.solver.geometry import Dimensions, Posture
from app.solver.placement import decode_chromosome
from app.solver.fitness import calculate_fitness, FitnessResult
from app.solver.sa import simulated_annealing, run_simulated_annealing


def _customer_group_bounds(units: List[Box]) -> List[Tuple[int, int]]:
    """For each unit index, return the (start, end) index range -- in the
    *current* units ordering -- of the contiguous block of units sharing
    its customer_sequence.

    resort_after_blocks() groups all units by customer_sequence into
    contiguous runs (descending sequence: last-unloaded customer placed
    deepest). check_lifo() only checks pairwise ordering at placement time,
    not after explode_blocks() re-expands blocks into individual cartons --
    so if the order genes let a unit "jump" across a customer boundary, two
    different customers' cargo can interleave in a way that violates strict
    unload-without-disturbing-other-customers LIFO, even though each
    individual placement passed check_lifo() when it was placed. Keeping
    every order-gene perturbation *within* its own customer's contiguous
    run preserves the customer separation that made the per-placement LIFO
    check sufficient in the first place, while still letting the GA/SA
    freely reorder units within a customer's own cargo (and freely reorder
    everything for FCL, which has a single implicit customer group).
    """
    n = len(units)
    bounds = [(0, n)] * n
    if n == 0:
        return bounds

    start = 0
    seqs = [getattr(u, 'customer_sequence', 0) for u in units]
    for i in range(1, n + 1):
        if i == n or seqs[i] != seqs[start]:
            for k in range(start, i):
                bounds[k] = (start, i)
            start = i
    return bounds


def _distinct_bounds(bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Collapse a per-index bounds list into the distinct (lo, hi) ranges,
    in order of first appearance."""
    seen = []
    last = None
    for b in bounds:
        if b != last:
            seen.append(b)
            last = b
    return seen


@dataclass
class Individual:
    chromosome: List[int]
    # Order genes: real-valued priority keys, one per unit, in the same
    # index space as `chromosome` / `units`. The placement *sequence* used
    # by decode_chromosome is argsort(order_keys) rather than the raw
    # 0..n-1 index order. Initialised to the identity permutation (i.e. the
    # order produced by initial_sort/resort_after_blocks) so an individual
    # with untouched order_keys behaves exactly like the original
    # index-order-only chromosome. GA crossover/mutation and SA neighbor
    # moves can perturb these keys to explore reorderings, while a small
    # perturbation magnitude keeps individuals close to the heuristic
    # ordering (preserving customer/LIFO grouping and item-affinity
    # clustering from sorting.py) rather than scrambling it outright.
    order_keys: List[float] = None
    fitness_result: Optional[FitnessResult] = None
    placed_bboxes: List = None
    placed_data: List = None
    unplaced: List = None
    current_weight: float = 0.0
    placed_postures: Optional[List[Posture]] = None

    def placement_order(self, units: Optional[List[Box]] = None) -> List[int]:
        """Return indices into `units`/`chromosome` in placement sequence.

        When `units` is given, the returned order is first clamped so no
        unit crosses its own customer_sequence group boundary (see
        _customer_group_bounds). This is enforced here -- once, at the one
        place order_keys actually gets turned into a sequence -- rather
        than in every crossover/mutation/SA operator individually, so the
        LCL customer-separation/LIFO invariant holds regardless of how the
        keys were perturbed upstream. Without `units` (e.g. legacy callers)
        this falls back to a plain unclamped argsort.
        """
        n = len(self.chromosome)
        if not self.order_keys:
            return list(range(n))

        if units is None:
            return sorted(range(len(self.order_keys)), key=lambda i: self.order_keys[i])

        bounds = _customer_group_bounds(units)
        order: List[int] = []
        for lo, hi in _distinct_bounds(bounds):
            group = sorted(range(lo, hi), key=lambda i: self.order_keys[i])
            order.extend(group)
        return order

    def clone(self) -> "Individual":
        """Fast shallow clone for GA/SA population operations.

        Perf note: profiling showed copy.deepcopy(individual) dominating GA
        runtime (rank-based selection alone was ~40% of total wall time on
        a 150-box LCL case) because it deep-copies placed_bboxes/placed_data
        /unplaced -- large nested structures that are fully *recomputed* by
        evaluate_individual() on essentially every use anyway. A shallow
        copy is safe here: `chromosome` and `order_keys` are always
        replaced wholesale (never mutated element-by-element in place) by
        mutate()/crossover(), and the placement result lists
        (placed_bboxes/placed_data/unplaced/placed_postures) are likewise
        always reassigned wholesale by evaluate_individual() rather than
        mutated in place -- the one place that *does* mutate a bbox list
        in place (compaction.run_compaction_pass) explicitly deep-copies
        its input first, specifically because it can't rely on this
        invariant. New list objects are created for chromosome/order_keys
        so mutating the clone can never affect the parent.
        """
        return Individual(
            chromosome=list(self.chromosome),
            order_keys=list(self.order_keys) if self.order_keys is not None else None,
            fitness_result=self.fitness_result,
            placed_bboxes=self.placed_bboxes,
            placed_data=self.placed_data,
            unplaced=self.unplaced,
            current_weight=self.current_weight,
            placed_postures=self.placed_postures,
        )


def create_individual(units: List[Box], order_jitter: float = 0.0) -> Individual:
    chromosome = []
    for unit in units:
        if unit.permitted_postures:
            idx = random.randrange(len(unit.permitted_postures))
        else:
            idx = 0
        chromosome.append(idx)

    n = len(units)
    if order_jitter > 0:
        # Identity order (0, 1, 2, ...) plus small random jitter so the
        # resulting argsort is a local perturbation of the heuristic sort
        # rather than a uniform-random shuffle. This keeps the population
        # concentrated near the (already good) heuristic ordering while
        # still letting GA/SA explore nearby permutations. Jitter is capped
        # so it can never push a unit's key outside its own customer's
        # contiguous index range (see _customer_group_bounds) -- this is
        # what keeps LCL customer separation / LIFO intact under
        # perturbation; for FCL every unit shares one group so this is a
        # no-op there.
        bounds = _customer_group_bounds(units)
        order_keys = []
        for i in range(n):
            lo, hi = bounds[i]
            jitter = min(order_jitter, (i - lo), (hi - 1 - i))
            jitter = max(0.0, jitter)
            order_keys.append(i + random.uniform(-jitter, jitter))
    else:
        order_keys = [float(i) for i in range(n)]

    return Individual(chromosome=chromosome, order_keys=order_keys)


def evaluate_individual(
    individual: Individual,
    units: List[Box],
    container_dims: Dimensions,
    max_weight: float,
    is_lcl: bool,
) -> Individual:
    order = individual.placement_order(units)
    ordered_units = [units[i] for i in order]
    ordered_chromosome = [individual.chromosome[i] for i in order]
    placed_bboxes, placed_data, unplaced, current_weight, placed_postures = decode_chromosome(
        ordered_chromosome, ordered_units, container_dims, max_weight, is_lcl
    )
    for idx, orig_i in enumerate(order):
        individual.chromosome[orig_i] = ordered_chromosome[idx]
    fitness_result = calculate_fitness(
        placed_bboxes, placed_data, unplaced, container_dims, max_weight
    )
    individual.fitness_result = fitness_result
    individual.placed_bboxes = placed_bboxes
    individual.placed_data = placed_data
    individual.unplaced = unplaced
    individual.current_weight = current_weight
    individual.placed_postures = placed_postures
    return individual


def rank_based_selection(population: List[Individual], count: int) -> List[Individual]:
    """Rank-based roulette selection: worst gets weight 1, best gets weight n."""
    ranked = sorted(population, key=lambda x: x.fitness_result.fitness if x.fitness_result else -float('inf'))
    n = len(ranked)
    weights = list(range(1, n + 1))
    total_weight = sum(weights)
    
    selected = []
    for _ in range(count):
        r = random.uniform(0, total_weight)
        cumulative = 0
        for i, ind in enumerate(ranked):
            cumulative += weights[i]
            if cumulative >= r:
                selected.append(ind.clone())
                break
    return selected


def crossover(parent1: Individual, parent2: Individual, crossover_probability: float) -> Tuple[Individual, Individual]:
    """Multi-point crossover with 2 random points.

    Applies to both the posture chromosome and the order_keys array using
    the *same* crossover points, so a child's order genes and posture genes
    stay aligned to the same index segments. order_keys are real-valued, so
    a straightforward segment swap (rather than a permutation-safe crossover
    like OX/PMX) is valid here -- there is no "invalid permutation" state to
    repair since argsort() always yields a valid permutation regardless of
    the key values.
    """
    if random.random() >= crossover_probability:
        return parent1.clone(), parent2.clone()

    if len(parent1.chromosome) != len(parent2.chromosome) or len(parent1.chromosome) <= 1:
        return parent1.clone(), parent2.clone()

    length = len(parent1.chromosome)
    if length == 2:
        point_a, point_b = 1, 2
    else:
        point_a, point_b = sorted(random.sample(range(1, length), 2))

    child1_chrom = (
        parent1.chromosome[:point_a] +
        parent2.chromosome[point_a:point_b] +
        parent1.chromosome[point_b:]
    )
    child2_chrom = (
        parent2.chromosome[:point_a] +
        parent1.chromosome[point_a:point_b] +
        parent2.chromosome[point_b:]
    )

    p1_keys = parent1.order_keys or [float(i) for i in range(length)]
    p2_keys = parent2.order_keys or [float(i) for i in range(length)]

    child1_keys = p1_keys[:point_a] + p2_keys[point_a:point_b] + p1_keys[point_b:]
    child2_keys = p2_keys[:point_a] + p1_keys[point_a:point_b] + p2_keys[point_b:]

    child1 = Individual(chromosome=child1_chrom, order_keys=child1_keys)
    child2 = Individual(chromosome=child2_chrom, order_keys=child2_keys)

    return child1, child2


def mutate(
    individual: Individual,
    units: List[Box],
    mutation_rate: float,
    order_mutation_scale: float = 1.5,
) -> Individual:
    """Random mutation with dynamic rate.

    Posture genes: unchanged behaviour (resample a random permitted posture).
    Order genes: each selected gene's priority key is nudged by a bounded
    random offset (rather than resampled uniformly at random), so mutation
    produces local re-orderings -- e.g. swapping two adjacent units or
    shifting one unit a few positions -- instead of teleporting a unit to
    an arbitrary point in the sequence. This keeps mutated individuals close
    to the heuristic (customer/LIFO/item-affinity) ordering on average while
    still letting the search escape it when doing so improves fitness.
    """
    mutated = individual.clone()

    n = len(mutated.chromosome)
    if mutated.order_keys is None or len(mutated.order_keys) != n:
        mutated.order_keys = [float(i) for i in range(n)]

    for i in range(n):
        if random.random() < mutation_rate:
            if i < len(units) and units[i].permitted_postures:
                mutated.chromosome[i] = random.randrange(len(units[i].permitted_postures))
        if random.random() < mutation_rate:
            mutated.order_keys[i] += random.uniform(-order_mutation_scale, order_mutation_scale)

    return mutated


def genetic_algorithm(
    units: List[Box],
    container_dims: Dimensions,
    max_weight: float,
    is_lcl: bool,
    population_size: int = None,
    generations: int = None,
    elite_fraction: float = None,
    crossover_probability: float = None,
    mutation_rate_base: float = None,
    mutation_rate_max: float = None,
    mutation_rate_min: float = None,
    sa_interval: int = None,
    min_improvement: float = None,
    early_stop_patience: int = None,
    progress_callback: Callable[[int, Individual], None] = None,
) -> Individual:
    settings = get_settings()

    pop_size = population_size or settings.POPULATION_SIZE
    num_gens = generations or settings.GENERATIONS
    elite_frac = elite_fraction or settings.ELITE_FRACTION
    cross_prob = crossover_probability or settings.CROSSOVER_PROBABILITY
    mut_base = mutation_rate_base or settings.MUTATION_RATE_BASE
    mut_max = mutation_rate_max or settings.MUTATION_RATE_MAX
    mut_min = mutation_rate_min or settings.MUTATION_RATE_MIN
    sa_int = sa_interval or settings.SA_INTERVAL_GENERATIONS
    min_imp = min_improvement or settings.MIN_IMPROVEMENT
    patience = early_stop_patience or settings.EARLY_STOP_PATIENCE
    # BUG-16 fix: hoist settings out of the per-generation loop
    # (lru_cache means it's cheap, but saving the lookup inside hot loops adds up)
    _min_imp = min_imp

    # order_jitter seeds initial diversity in placement order around the
    # heuristic sort (see create_individual). A modest jitter keeps most of
    # the population close to the proven initial_sort/resort_after_blocks
    # ordering while still giving the GA a range of starting sequences to
    # select from, rather than every individual starting from the identical
    # order and only posture varying (the original behaviour).
    population = [create_individual(units, order_jitter=3.0) for _ in range(pop_size)]

    for ind in population:
        evaluate_individual(ind, units, container_dims, max_weight, is_lcl)

    population.sort(key=lambda x: x.fitness_result.fitness if x.fitness_result else -float('inf'), reverse=True)

    best_individual = population[0].clone()
    best_fitness = best_individual.fitness_result.fitness if best_individual.fitness_result else -float('inf')
    stagnant_generations = 0
    mutation_rate = mut_base

    for gen in range(1, num_gens + 1):
        fitnesses = [ind.fitness_result.fitness if ind.fitness_result else -float('inf') for ind in population]
        gen_best_idx = fitnesses.index(max(fitnesses))

        # Check for meaningful improvement
        if fitnesses[gen_best_idx] > best_fitness + min_imp:
            best_individual = population[gen_best_idx].clone()
            best_fitness = fitnesses[gen_best_idx]
            stagnant_generations = 0
            mutation_rate = max(mut_min, mutation_rate * 0.9)
        else:
            stagnant_generations += 1
            mutation_rate = min(mut_max, mutation_rate * 1.1)
            # Keep sub-threshold improvements but don't reset patience
            if fitnesses[gen_best_idx] > best_fitness:
                best_individual = population[gen_best_idx].clone()
                best_fitness = fitnesses[gen_best_idx]

        # Simulated Annealing as local operator
        if gen % sa_int == 0:
            # BUG-09 fix: pass the real max_weight instead of volume*0.001
            sa_individual, sa_fitness = run_simulated_annealing(
                container_dims, units, best_individual, best_fitness, is_lcl,
                max_weight=max_weight,
            )
            if sa_fitness > best_fitness + min_imp:
                best_individual = sa_individual
                best_fitness = sa_fitness
                stagnant_generations = 0
            elif sa_fitness > best_fitness:
                best_individual = sa_individual
                best_fitness = sa_fitness

        # Early stopping: if all units are placed and stagnant for 3 generations, or if stagnant for patience
        all_placed = bool(best_individual.placed_data and len(best_individual.placed_data) == len(units))
        if (all_placed and stagnant_generations >= 3) or stagnant_generations >= patience:
            break

        # Selection and reproduction
        elite_count = max(1, int(pop_size * elite_frac))
        elites = population[:elite_count]
        selected = rank_based_selection(population, pop_size - elite_count)

        next_generation = [e.clone() for e in elites]

        while len(next_generation) < pop_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child1, child2 = crossover(parent1, parent2, cross_prob)

            child1 = mutate(child1, units, mutation_rate)
            child2 = mutate(child2, units, mutation_rate)

            evaluate_individual(child1, units, container_dims, max_weight, is_lcl)
            evaluate_individual(child2, units, container_dims, max_weight, is_lcl)

            next_generation.append(child1)
            if len(next_generation) < pop_size:
                next_generation.append(child2)

        population = next_generation[:pop_size]
        population.sort(key=lambda x: x.fitness_result.fitness if x.fitness_result else -float('inf'), reverse=True)

        if progress_callback:
            progress_callback(gen, best_individual)

    return best_individual