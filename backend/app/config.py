from typing import Tuple
from pydantic_settings import BaseSettings
from functools import lru_cache



# Top-corner casting obstruction dimensions (ISO 1161 approximation: 17.8 x 16.2 x 11.8 cm)
CORNER_BLOCK_X_CM: float = 17.8
CORNER_BLOCK_Y_CM: float = 16.2
CORNER_BLOCK_Z_CM: float = 11.8


class Settings(BaseSettings):
    # API
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "3D Container Loading DSS"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # Solver Parameters (Section 7)
    # Genetic Algorithm
    POPULATION_SIZE: int = 60
    GENERATIONS: int = 100
    ELITE_FRACTION: float = 0.10
    MUTATION_RATE_BASE: float = 0.25
    MUTATION_RATE_MAX: float = 0.50
    MUTATION_RATE_MIN: float = 0.10
    CROSSOVER_PROBABILITY: float = 0.7
    MIN_IMPROVEMENT: float = 0.01
    EARLY_STOP_PATIENCE: int = 60

    # Simulated Annealing (embedded in GA)
    SA_INTERVAL_GENERATIONS: int = 5
    SA_INITIAL_TEMP: float = 100.0
    SA_MIN_TEMP: float = 1.0
    SA_COOLING_RATE: float = 0.9

    # Block Generation
    MIN_BLOCK_FILL_RATIO: float = 0.75
    MAX_BLOCK_FRACTION_X: float = 0.30  # GA search cap — length/X axis (unchanged)
    MAX_BLOCK_FRACTION_Y: float = 0.80  # GA search cap — width/Y axis (allows 2-3 carton layers)
    MAX_BLOCK_FRACTION_Z: float = 0.80 # GA search cap — height/Z axis (allows 2-3 carton tiers)
    MAX_BLOCK_FRACTION_REPORT: float = 0.95  # Report/visual cap (>= search cap)
    SIMILAR_SIZE_TOLERANCE: float = 0.1
    BLOCK_HEIGHT_DEADZONE: Tuple[float, float] = (0.35, 0.75)

    # Strategy Variants
    USE_STATIC_BLOCKS: bool = False
    GROUP_KEY: str = "geometry"  # "item_id" | "geometry"
    GA_LEVEL: str = "group"    # "carton" | "group"
    DYNAMIC_BLOCKS: bool = True
    POST_EXPLODE_COMPACTION: bool = True


    # Placement Strategy
    SORT_ITEM_AFFINITY: bool = False
    WALL_FIRST_RULE: str = "B"  # "A" (baseline), "B" (strict wall-first, validated +4.3% gain), "C" (soft overrun penalty)
    WALL_FIRST_WEIGHT: float = 1.0  # Weight w for Rule C: score - w * (overrun / L)
    WALL_FIRST_SECONDARY_ORDER: str = "ZY"  # "ZY" (lower z then smaller y) or "YZ" (smaller y then lower z)
    WALL_FIRST_PENALTY: float = 2.0  # Overrun penalty for FCL dynamic blocks branch
    TOLERANCE_GAP_CM: float = 0.0
    SUPPORT_RATIO: float = 0.6
    CONTACT_RATIO_WEIGHT: float = 1.0
    RESIDUAL_VOLUME_WEIGHT: float = 1.0

    # Constraints
    MAX_WEIGHT_UTILIZATION: float = 1.0
    COG_TOLERANCE_XY: float = 0.05  # ±5% of length/width
    COG_TOLERANCE_Z: float = 0.10   # +10% of height
    UNPLACED_RANK_WEIGHT: float = 2.0
    CORNER_BLOCK_X_CM: float = CORNER_BLOCK_X_CM
    CORNER_BLOCK_Y_CM: float = CORNER_BLOCK_Y_CM
    CORNER_BLOCK_Z_CM: float = CORNER_BLOCK_Z_CM

    # Fitness weights
    FITNESS_VOLUME_WEIGHT: float = 1.0
    FITNESS_COG_PENALTY_WEIGHT: float = 0.3  # cog_weight
    FITNESS_FRAG_PENALTY_WEIGHT: float = 0.5  # anti-fragmentation weight omega_frag (exposed frontier)
    # INFEASIBLE_PENALTY computed per run

    # Default container (used if none selected)
    DEFAULT_CONTAINER_TYPE: str = "40HC"
    DEFAULT_CONTAINER_LENGTH: float = 1203.2
    DEFAULT_CONTAINER_WIDTH: float = 235.2
    DEFAULT_CONTAINER_HEIGHT: float = 270.0
    DEFAULT_CONTAINER_MAX_WEIGHT: float = 28000.0

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()