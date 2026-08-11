# Current run artifact and precipitation replacements

## Goal

Make the pre-run SQLite artifact a replaceable cache of the current run. It must
tell the MGB preparation pipeline which observations to replace and which
stations to omit without modifying or duplicating the observations stored in
`history.sqlite`.
Long-lived run snapshots are optional rather than an automatic product of every
execution.

## Requirements

- Maintain one current run artifact at
  `<workspace>/data/cache/current_run.sqlite`. Create it before the first MGB
  preparation and update that artifact in place, transactionally, as the user
  changes the current run.
- Treat the current artifact as the source of effective run settings, enabled
  observed providers, interpolation settings, review window, forecast scenario
  references, replacement rules, and operator decisions.
- Do not copy observed series or values from `history.sqlite` into the run
  artifact. Legacy run-input tables may remain for compatibility but must not be
  populated by this workflow.
- Do not write flags, exclusions, curated observations, or station status to
  `history.sqlite`. Its existing `qc_flag` structure becomes inactive for this
  workflow and raw provider observations remain unchanged.
- Keep flags out of the run artifact. Flags are a dashboard visualization of
  suspect observations, not persisted run data.
- Support these initial default replacement criteria, using the normalized run
  timestep:
  - replace values strictly greater than `50 mm` with `NULL`, which removes that
    station value from interpolation for the timestep;
  - replace with `0.0 mm` every value in a sequence of at least five consecutive
    timesteps when every value is strictly greater than `0 mm` and strictly less
    than `1 mm`;
  - break a sequence on a missing timestep or an out-of-band value, and do not
    treat ordinary zero-rain sequences as suspect.
- Accept the replacement criteria explicitly through the library API, with
  validated regional defaults that callers may override.
- When first constructing the current artifact, create the default replacement
  for every observation that matches the effective criteria. Users may change
  or remove a replacement, or add another replacement, by updating the artifact.
- Store value replacements as `station_id + observed_at + replacement_value`.
  The replacement may be a finite non-negative precipitation value or `NULL`;
  `NULL` is equivalent to excluding that value from interpolation. Removing the
  replacement instruction restores use of the original history value. Store
  station exclusions as `station_id + (start, end]` instructions.
- Attribute responsibility once at run level, not on individual values,
  sequences, or station periods. A non-empty run-level reason is mandatory;
  initialize the first artifact with `default precipitation replacements`.
- Validate every update and apply it atomically while preserving the stable
  current-artifact path. The same artifact may be edited and executed repeatedly.
- Executing an artifact must query the latest preferred rainfall series from
  `history.sqlite`; changes to history after preparation do not invalidate the
  artifact.
- Apply value replacements and station-window exclusions consistently to every observed
  precipitation interpolation path. If an effective timestep has no usable
  stations, fail preparation rather than silently substituting rainfall.
- Build the observed working-grid cache once per execution and reuse it across
  all forecast scenarios so every scenario receives identical observed forcing.
- Keep only the current execution status and current published asset references
  in the cache artifact; do not accumulate an unbounded execution history.
- Preserve the existing all-or-nothing cache publication behavior: a failed
  execution records the current failure but leaves the last published caches
  intact.
- Make persistent storage opt-in. Saving the current run copies it to
  `<workspace>/data/runs/<run_id>.sqlite` and requires an explicit, unique
  `run_id` and non-empty description. The saved artifact retains its run-level
  responsible person and reason; ordinary Refresh executions do not create
  persistent run files.
- Expose this behavior through structured, importable Python contracts and
  workflow functions. No HTTP or REST API is part of this change.

## Notes

- The current run artifact is a disposable instruction cache, not a closed copy
  of history data or an execution archive.
- Suspect-value detection and default replacement selection may be shared as a
  reusable policy evaluator, but `flag` labels and state belong exclusively to
  dashboard presentation.
- A workspace-level execution lock is required because scenario and observed
  cache publication targets are shared between dashboard sessions.
- This replaces the documented direction of persistent observation QC in
  history; persistent project documentation must be updated when implementation
  is complete.

## Verification

- Boundary tests cover `50 mm`, values above `50 mm`, strict sequence bounds at
  `0 mm` and `1 mm`, five-step and longer sequences, gaps, and zero runs.
- Tests prove threshold matches default to `NULL`, sequence matches default to
  `0.0`, `NULL` removes only that station/timestep from interpolation, numeric
  replacements are interpolated, and removing an instruction restores the
  history value.
- Run-artifact tests prove that no observed values or dashboard flags are
  copied, run-level responsibility and reason are enforced, the initial reason
  defaults correctly, and in-place updates are atomic.
- Integration tests prove execution uses current history, applies the same
  observed forcing to every scenario, replaces current execution metadata,
  rejects uncovered timesteps, and preserves published caches after failure.
- Persistence tests prove normal execution creates no archived run and that an
  explicit save requires `run_id` and description and leaves the current cache
  available for further updates.
