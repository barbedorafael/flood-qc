# Dashboard observed precipitation review and diagnostics

## Goal

Expose current-run observed-precipitation filtering in a dedicated dashboard
tab and give users spatial tools to understand suspect values, effective
replacements, station availability, and leave-one-out interpolation before
updating and running MGB.

## Requirements

- Add an `Observed Precipitation QC` top-level tab backed by session-local
  dashboard state. Draft scans, flags, exclusions, station selections, and
  diagnostics must not write to `history.sqlite`; only an explicit update writes
  the current run cache.
- Provide review-window controls and editable QC parameters seeded from the
  validated regional defaults defined by the run-artifact change.
- Derive flags only for dashboard filtering and visualization. Do not persist
  them in the current or saved run artifacts.
- Present the derived flags in a filterable table and spatial station view,
  distinguishing threshold flags, sequence flags, replaced values,
  and stations excluded for the selected window.
- Allow users to set or remove a replacement for an individual flagged value or
  a complete detected sequence. Threshold matches initially use `NULL`, and
  sequences strictly between `0 mm` and `1 mm` initially use `0.0`; users may
  enter another finite non-negative value or remove the instruction to restore
  the history value.
- Allow a station to be excluded only for the selected `(start, end]` review
  window; this is not a persistent station enablement setting.
- Provide one responsible-person field and one mandatory reason for the current
  run rather than audit fields on individual values or periods. Seed the first
  run reason with `default precipitation replacements` and allow editing it.
- Provide accumulated leave-one-out diagnostics for a selected station and
  review window:
  - omit the selected station from every interpolation timestep;
  - apply current draft replacements and station masks to all other sources;
  - use the same IDW nearest-station count and power as production;
  - display the accumulated interpolated field and the omitted station on the
    map;
  - report observed accumulation, predicted accumulation, signed residual
    `predicted - observed`, absolute error, and timestep coverage;
  - retain the selected station's raw observations only as diagnostic truth and
    require complete target coverage before reporting a difference.
- Add an explicit `Update Current Run` action that validates the session draft
  and transactionally creates or updates
  `<workspace>/data/cache/current_run.sqlite` through the library API. The draft
  remains editable and later updates reuse the same artifact.
- Add an optional `Save Run` action for long-lived storage. It requires a unique
  run ID and non-empty description and must not be invoked automatically by
  Update or Refresh.
- Change the dashboard Refresh action so it requires and confirms a valid
  current artifact, then runs the preparation/MGB/export pipeline from that
  artifact rather than merely re-reading existing files.
- Run the pipeline without blocking dashboard progress/error feedback and
  prevent concurrent execution for the same workspace.
- On success, republish all caches, include the new execution generation in
  cached-loader keys, and reload the browser page completely. On failure, show
  the recorded error without reloading and continue displaying the last valid
  caches.
- After reload, restore controls and effective replacements directly from the
  stable current-artifact path.
- Keep domain checks, run-artifact persistence, interpolation, and pipeline
  orchestration in reusable library modules. The Panel application owns flag
  state, session state, callbacks, presentation, and cache/reload behavior.

## Notes

- Refresh does not download new provider data; it rebuilds MGB inputs, executes
  the current scenario batch against the latest history observations, exports
  outputs, republishes caches, and reloads the dashboard.

## Verification

- UI tests cover session isolation, editable defaults, non-persistent flag
  filtering and map state, default `NULL` and zero replacements, custom numeric
  replacements, replacement removal, station-window exclusions, and run-level
  responsibility and reason.
- Analysis tests cover accumulated leave-one-out values, target omission, draft
  masks, residual sign, incomplete coverage, and insufficient remaining
  stations.
- Workflow/UI tests cover repeated in-place updates, optional persistent save,
  required save identifiers and description, Refresh confirmation, progress and
  failure feedback, execution locking, cache-generation changes, current-artifact
  recovery, and full-page reload after success.
