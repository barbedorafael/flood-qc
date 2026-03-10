PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS run_metadata (
    run_id TEXT PRIMARY KEY,
    reference_time TEXT NOT NULL,
    run_kind TEXT NOT NULL CHECK (run_kind IN ('automatic', 'manual')),
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'ready', 'executed', 'reviewed', 'published')),
    parent_run_id TEXT,
    operator TEXT,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS run_lineage (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES run_metadata(run_id) ON DELETE CASCADE,
    parent_run_id TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'derived_from'
);

CREATE TABLE IF NOT EXISTS input_series (
    id INTEGER PRIMARY KEY,
    series_key TEXT NOT NULL UNIQUE,
    variable TEXT NOT NULL,
    unit TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'raw' CHECK (state IN ('raw', 'curated', 'approved')),
    source_ref TEXT
);

CREATE TABLE IF NOT EXISTS input_value (
    id INTEGER PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES input_series(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    value REAL,
    UNIQUE (series_id, observed_at)
);

CREATE TABLE IF NOT EXISTS model_execution (
    id INTEGER PRIMARY KEY,
    runner_name TEXT NOT NULL DEFAULT 'mgb',
    planned_command TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    executed_at TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS output_series (
    id INTEGER PRIMARY KEY,
    series_key TEXT NOT NULL UNIQUE,
    variable TEXT NOT NULL,
    unit TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'raw' CHECK (state IN ('raw', 'curated', 'approved')),
    source_ref TEXT
);

CREATE TABLE IF NOT EXISTS output_value (
    id INTEGER PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES output_series(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    value REAL,
    UNIQUE (series_id, observed_at)
);

CREATE TABLE IF NOT EXISTS qc_flag (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    reference_id TEXT NOT NULL,
    rule_code TEXT NOT NULL,
    severity TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open',
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manual_edit (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT NOT NULL,
    editor TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS asset_ref (
    id INTEGER PRIMARY KEY,
    asset_kind TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'raw' CHECK (state IN ('raw', 'curated', 'approved')),
    description TEXT
);

CREATE TABLE IF NOT EXISTS report_artifact (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_input_value_series_time ON input_value(series_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_output_value_series_time ON output_value(series_id, observed_at);