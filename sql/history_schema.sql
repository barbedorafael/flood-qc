PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS station (
    id INTEGER PRIMARY KEY,
    station_code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    station_type TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    operator TEXT NOT NULL CHECK (operator IN ('ana', 'sgb', 'inmet'))
);

CREATE TABLE IF NOT EXISTS observed_series (
    id INTEGER PRIMARY KEY,
    station_id INTEGER NOT NULL REFERENCES station(id) ON DELETE CASCADE,
    variable TEXT NOT NULL,
    unit TEXT NOT NULL,
    source_name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'raw' CHECK (state IN ('raw', 'curated', 'approved')),
    source_path TEXT,
    UNIQUE (station_id, variable, source_name, state)
);

CREATE TABLE IF NOT EXISTS observed_value (
    id INTEGER PRIMARY KEY,
    series_id INTEGER NOT NULL REFERENCES observed_series(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    value REAL,
    quality_code TEXT,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
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

CREATE TABLE IF NOT EXISTS external_asset (
    id INTEGER PRIMARY KEY,
    asset_kind TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'raw' CHECK (state IN ('raw', 'curated', 'approved')),
    description TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS run_catalog (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    run_kind TEXT NOT NULL CHECK (run_kind IN ('automatic', 'manual')),
    parent_run_id TEXT,
    reference_time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'ready', 'executed', 'reviewed', 'published')),
    run_db_path TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_observed_value_series_time ON observed_value(series_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_run_catalog_status ON run_catalog(status);