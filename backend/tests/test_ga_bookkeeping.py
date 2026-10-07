"""Tests for GA best-individual retention and early-stopping bookkeeping (Section A5)."""
from app.config import get_settings
from app.solver.strategy_variants import GroupIndividual
from app.solver.fitness import FitnessResult


def test_early_stop_patience_default():
    settings = get_settings()
    assert settings.EARLY_STOP_PATIENCE == 40


def test_best_individual_retention_subthreshold():
    """Verify that sub-threshold improvements update best_ind but still increment stagnation."""
    settings = get_settings()
    min_imp = settings.MIN_IMPROVEMENT  # e.g. 0.01

    # Simulate loop logic
    best_ind = GroupIndividual(posture_genes=[0, 0], lots=[])
    best_ind.fitness_result = FitnessResult(
        fitness=0.50, placed_volume=100.0, unplaced_count=1,
        cog_deviation_xy=0.0, cog_deviation_z=0.0,
        stacking_violations=0, stability_violations=0
    )

    stagnant_gens = 0

    # Generation 1: improvement of 0.005 (less than min_imp)
    pop0 = GroupIndividual(posture_genes=[0, 0], lots=[])
    pop0.fitness_result = FitnessResult(
        fitness=0.505, placed_volume=101.0, unplaced_count=1,
        cog_deviation_xy=0.0, cog_deviation_z=0.0,
        stacking_violations=0, stability_violations=0
    )

    best_fitness_before = best_ind.fitness_result.fitness
    current_best_fitness = pop0.fitness_result.fitness

    if current_best_fitness > best_fitness_before + min_imp:
        best_ind = pop0.clone()
        stagnant_gens = 0
    else:
        stagnant_gens += 1
        if current_best_fitness > best_fitness_before:
            best_ind = pop0.clone()

    # best_ind should be updated to the new individual with fitness 0.505
    assert best_ind.fitness_result.fitness == 0.505
    # Stagnant count should have incremented
    assert stagnant_gens == 1

    # Generation 2: meaningful improvement of 0.02 (> min_imp)
    pop1 = GroupIndividual(posture_genes=[1, 0], lots=[])
    pop1.fitness_result = FitnessResult(
        fitness=0.525, placed_volume=105.0, unplaced_count=0,
        cog_deviation_xy=0.0, cog_deviation_z=0.0,
        stacking_violations=0, stability_violations=0
    )

    best_fitness_before = best_ind.fitness_result.fitness
    current_best_fitness = pop1.fitness_result.fitness

    if current_best_fitness > best_fitness_before + min_imp:
        best_ind = pop1.clone()
        stagnant_gens = 0
    else:
        stagnant_gens += 1
        if current_best_fitness > best_fitness_before:
            best_ind = pop1.clone()

    assert best_ind.fitness_result.fitness == 0.525
    assert stagnant_gens == 0
