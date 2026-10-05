"""Legacy simulated annealing module.

Moved to app.solver.legacy.sa. Re-exported here for backward compatibility.
"""
from app.solver.legacy.sa import (
    simulated_annealing,
    run_simulated_annealing,
    generate_neighbor,
)

__all__ = [
    "simulated_annealing",
    "run_simulated_annealing",
    "generate_neighbor",
]