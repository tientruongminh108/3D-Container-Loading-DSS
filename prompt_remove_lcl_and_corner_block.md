# Prompt for coding agent — 2 independent tasks

Base commit: `4b66c41`. Do these as two separate, independently verifiable changes.
Do not mix their diffs.

---

## TASK 1: Remove LCL-specific logic from the solver core (keep API/FE untouched)

**Goal:** the solver core must always behave as if every shipment is FCL — no
customer separation, no LIFO, no customer_sequence-aware ordering or placement
logic anywhere in `app/solver/`. This is a deliberate, already-confirmed product
decision: when a packing list still contains multiple distinct `Customer_Code`
values, the solver must simply treat all cartons as one undifferentiated batch and
pack for maximum fill, exactly like current FCL behavior. Do NOT add any
rejection/validation error for multi-customer input — just ignore customer identity
entirely in the core.

**Scope:** `app/solver/` only. Do NOT touch:
- `app/core/models.py` (Pydantic schemas: `ShipmentType`, `customer_sequence`
  fields, etc. stay as-is — API/FE contracts must not break)
- Any FastAPI route/endpoint code
- Frontend code
- `app/services/mock_packer.py` — leave as-is for now; confirm separately whether
  it still needs LCL handling for API/demo purposes, don't touch it in this task

### Known locations of `is_lcl` / LCL / `customer_sequence` branching logic to remove
(verified by grep against the current codebase — treat this as a starting map,
re-grep after your edits to confirm nothing is missed):

- **`app/solver/pipeline.py`**: `is_lcl = shipment_type == ShipmentType.LCL` and the
  auto-strategy-selection branch (currently: LCL → Variant B static blocks +
  compaction, FCL → Variant E_PEC dynamic blocks). See ADDENDUM below for exactly
  what must replace this branch. Remove the `is_lcl` parameter threading through
  `run_pipeline`'s internal calls once nothing downstream needs it.
- **`app/solver/constraints.py`**: `check_lifo()` function and its call site —
  remove the function and the `if is_lcl and not check_lifo(...)` branch in
  `placement.py`.
- **`app/solver/placement.py`**: `is_lcl` parameter on `find_best_placement`,
  `place_boxes_greedy`, `place_blocks_greedy`, `decode_chromosome`; the
  `max_corners = 2 if is_lcl else 4` logic (should just always be 4); the LCL/FCL
  corner-seeding branch.
- **`app/solver/compaction.py`**: `is_lcl` parameter threaded through
  `compact_x_rear`, `compact_y_sidewall`, `compact_z_downward`,
  `run_compaction_pass`, and `_lifo_ok_after_move()` — remove the LIFO re-check
  during compaction entirely.
- **`app/solver/strategy_variants.py`**: `GroupIndividual.ordered_lots(groups,
  is_lcl)` — simplify to always behave as the `is_lcl=False` branch; remove the
  `wall_penalty = ... if not is_lcl else 10.0` special case; remove the
  `if is_lcl and not check_lifo(...)` candidate-grid filter.
- **`app/solver/ga.py`** and **`app/solver/sa.py`**: `is_lcl` parameter threading
  into `evaluate_individual` — trace whether `evaluate_individual` itself branches
  on it (check `fitness.py` too) or just passes it through to placement calls being
  removed above; drop the parameter if it becomes a no-op.
- **`app/solver/sorting.py`**: the `-customer_sequence` component of sort keys and
  the `shipment_type == "LCL"` checks — customer_sequence should no longer affect
  sort order; simplify keys to `(-volume, -weight, item_id, box_id)` consistently.
- **`app/solver/output.py`**: `shift_downward_cartons(is_lcl=...)` and
  `build_unplaced_cartons(..., is_lcl)` — check what LCL-specific behavior these
  have and remove it.
- **`app/solver/validator.py`**: the `if is_lcl:` branch (~line 169) that
  presumably runs the LIFO check — remove the LIFO validation entirely from the
  core validator, since there is no longer any LIFO concept to validate. Keep the
  `is_lcl` parameter on `validate_solution`'s signature ONLY IF something outside
  `app/solver/` (API layer) depends on that function signature — check callers
  first; if `validate_solution` is only called from within `app/solver/` and
  `benchmark_runner.py` (a dev tool, not product API), it's safe to drop the
  parameter too. If unsure, leave the parameter but make it a no-op inside, and
  flag this ambiguity in your summary.
- **`app/solver/parsing.py`**: `detect_shipment_type()` currently returns
  `(ShipmentType.LCL, customer_count, customer_sequence_map)` when multiple
  `Customer_Code` values are present. Decide: either keep this function as-is
  (since it only produces data, doesn't enforce LCL behavior, and the core now
  ignores `shipment_type` for strategy purposes) OR simplify it to always return
  `ShipmentType.FCL` / `cust_seq=0` for everything if nothing in the core reads the
  LCL value anymore. Prefer keeping it as-is and just confirming nothing downstream
  in `app/solver/` branches on its output, since `parsing.py`'s job is arguably
  still "produce accurate shipment_type data" even if solver no longer acts on it
  differently — use your judgment and document which you chose and why.

**Process:** after removing each `is_lcl` branch, grep the whole `app/solver/`
directory again for `"is_lcl"`, `"ShipmentType.LCL"`, `"check_lifo"`,
`"customer_sequence"` to confirm nothing was missed, and to decide case-by-case
whether a remaining match is dead code to delete or legitimate data-only usage to
keep.

### ADDENDUM — critical, do not skip

After removing the `is_lcl` branch in `pipeline.py` (the block that currently
chooses between "LCL → static blocks/Variant B" and "FCL → dynamic
blocks/Variant E_PEC"), you must explicitly decide what replaces it. **The correct
replacement is: always use the E_PEC configuration (the current FCL branch),
unconditionally, for every input regardless of how many distinct `Customer_Code`
values it has.** Concretely:

```python
use_static_blocks = False
dynamic_blocks = True
ga_level = "group"
group_key = "geometry"
post_explode_compaction = True
```

This was confirmed as the better-performing configuration (benchmark comparisons:
Variant E_PEC vs Variant B on FCL data — E_PEC achieves materially higher fill
rate) and the ONLY reason Variant B/static-blocks was ever used for LCL was to
protect LIFO customer-separation — which no longer exists in the core after this
task, so there is no remaining reason to keep the static-blocks path as a default
for any input.

Also update the defaults in `app/config.py` to match, since these are the fallback
values used whenever `options` is `None` or doesn't specify these fields — they
currently default to the OLD "Variant B" configuration
(`USE_STATIC_BLOCKS=True, GA_LEVEL="carton", DYNAMIC_BLOCKS=False`), which would
silently undo this change if left as-is:

```python
USE_STATIC_BLOCKS: bool = False
GROUP_KEY: str = "geometry"
GA_LEVEL: str = "group"
DYNAMIC_BLOCKS: bool = True
POST_EXPLODE_COMPACTION: bool = True
```

Do NOT delete the static-blocks code path itself
(`app/solver/block_generation.py`, the `if use_static_blocks:` branch in
`pipeline.py`, `place_blocks_greedy`, etc.) — it should remain reachable via
explicit `RunOptions` for benchmarking/debugging purposes via
`benchmark_runner.py` (which still legitimately needs to compare
A/B/C/D/E/*_PEC variants as a dev tool, used only by developers — never exposed
through the API or frontend). Only change what the system does by DEFAULT when
the caller doesn't explicitly request a specific strategy.

### Verification

1. Run the full test suite (`cd backend && python -m pytest tests/ -q`). Tests
   that specifically assert LCL/LIFO behavior will now fail or need removal —
   remove obsolete LCL-specific tests, but flag which ones you removed and why in
   your summary; don't silently delete tests that might be testing something else
   too.
2. Run `backend/benchmark_runner.py --datasets 01 --budgets interactive --variants
   A B --num-seeds 3 --workers 1` (dataset 01 is the 3-customer LCL dataset) and
   confirm: (a) it still runs without crashing even though it's multi-customer
   data, (b) `lifo_violations` is no longer a meaningful column (or is always 0
   since there's no LIFO check anymore) — don't worry about optimizing fill for
   this dataset, we're confirming the core treats it as one undifferentiated FCL
   batch, not validating LCL quality.
3. Run the same on dataset 03 (FCL) and confirm fill rate and valid rate are
   unchanged from before this change (should still match the most recent E_PEC
   baseline, ~100% valid) — this change must not regress FCL behavior.
4. Call `run_pipeline` with `options=None` (or `RunOptions()` with no strategy
   fields set) on both dataset 01 (3-customer) and dataset 03 (FCL) and confirm
   both use the dynamic-blocks/group-level path by default (e.g. by checking
   which code branch executes, or by confirming fill rate matches E_PEC benchmark
   numbers rather than old Variant B numbers for both).

---

## TASK 2: Model top-corner obstruction (corner casting) as excluded volume

**Goal:** real ISO shipping containers have a structural corner casting/corner
post at each of the 8 corners (4 top, 4 bottom); cargo cannot occupy the space
those structures intrude into at the top 4 corners of the usable cargo volume.
Currently `app/solver/parsing.py`'s `parse_container_spec()` computes
`usable_length/width/height` as a simple rectangular box (`Internal_*` minus
`2 * tolerance_gap` on length/width only). We need to additionally exclude a small
cuboid at each of the 4 TOP corners.

**Default dimensions** (not verified against a real spec for this exact fleet —
use as a configurable default, document it as an approximation based on ISO 1161
corner casting external envelope, 178×162×118mm): `17.8 cm × 16.2 cm × 11.8 cm`.
Make this a named constant in `app/config.py`:

```python
CORNER_BLOCK_X_CM: float = 17.8
CORNER_BLOCK_Y_CM: float = 16.2
CORNER_BLOCK_Z_CM: float = 11.8
```

(Axis meaning: X = length-direction intrusion, Y = width-direction intrusion,
Z = height-direction intrusion downward from the roof, at each of the 4 top
corners — rear-left, rear-right, door-left, door-right — using this codebase's
existing `x=0`-is-rear / `x=L`-is-door convention.)

### Implementation

This is **NOT** a simple reduction of `usable_length/width/height` — that would
shrink the box globally, which is wrong, since the obstruction is local to 4
small corner regions near the roof, not the whole volume. Instead:

1. Add a new constraint function in `app/solver/constraints.py`, e.g.
   `check_corner_clearance(candidate: BoundingBox, container_dims: Dimensions) -> bool`,
   that rejects a candidate box if it intersects any of the 4 top-corner excluded
   cuboids. Each cuboid's region (using `CORNER_BLOCK_X/Y/Z_CM` from settings):

   - **Rear-left-top:** `x in [0, CORNER_BLOCK_X_CM]`, `y in [0, CORNER_BLOCK_Y_CM]`, `z in [container_dims.height - CORNER_BLOCK_Z_CM, container_dims.height]`
   - **Rear-right-top:** `x in [0, CORNER_BLOCK_X_CM]`, `y in [container_dims.width - CORNER_BLOCK_Y_CM, container_dims.width]`, same z range
   - **Door-left-top:** `x in [container_dims.length - CORNER_BLOCK_X_CM, container_dims.length]`, `y in [0, CORNER_BLOCK_Y_CM]`, same z range
   - **Door-right-top:** `x in [container_dims.length - CORNER_BLOCK_X_CM, container_dims.length]`, `y in [container_dims.width - CORNER_BLOCK_Y_CM, container_dims.width]`, same z range

   A candidate intersects a corner cuboid if its bbox overlaps that cuboid's
   region on all 3 axes (standard AABB overlap test — reuse whatever overlap-test
   style already exists in `check_non_overlap` for consistency).

2. Call `check_corner_clearance` from `find_best_placement` (`placement.py`)
   alongside the existing `check_container_bounds` call, and from
   `is_valid_shift` in `compaction.py` so a compaction move can't shift a box
   into a corner region either.

3. Add a `corner_block` field or similar to `RunResult`/metrics if useful for the
   frontend to visualize later, but this is optional — not required for this task
   since FE is out of scope; just make sure the core constraint works correctly
   first.

4. This only matters for cargo near the very top of the container (z close to
   `container_dims.height`) and near the far X ends (rear or door) and far Y
   edges (sidewalls) simultaneously — most placements won't be affected. Make
   sure the check is cheap (early-exit on the z range check first, since most
   candidates won't be near the roof).

### Verification

1. Write a focused unit test (add to `backend/tests/`) that places a box whose
   bbox would occupy e.g. `x=[0,10]`, `y=[0,10]`, `z=[container_height-5,
   container_height]` and confirm `check_corner_clearance` rejects it (overlaps
   the rear-left-top exclusion cuboid), and confirm a box at `x=[50,60]`,
   `y=[50,60]`, same z range is accepted (clear of all 4 corners).
2. Run the full test suite to confirm no regression.
3. Run `backend/benchmark_runner.py --datasets 03 --budgets interactive
   --variants A B --num-seeds 3 --workers 1` and report the fill rate — some
   small drop vs. the current baseline is expected and correct here, since real
   usable volume is now slightly smaller; report the delta so we can confirm
   it's a small, reasonable reduction (a few tenths of a percent, not a large
   drop) given the corner blocks are tiny relative to total container volume.

---

## General notes for both tasks

- Keep the two tasks' diffs separable (e.g. two commits or clearly labeled
  sections in your summary) so each can be reviewed and verified independently.
- After both tasks, run the full existing backend test suite one more time and
  report pass/fail counts.
- Do not touch `app/solver/constraints.py::check_stackability`'s weight-hierarchy
  rule (heavier-on-lighter rejected, equal-or-lighter-on-heavier allowed) or the
  per-carton support-ratio logic fixed in the previous round of changes — both
  are out of scope for this prompt.
