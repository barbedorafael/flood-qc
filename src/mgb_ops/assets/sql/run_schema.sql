PRAGMA foreign_keys = ON;

-- This database is intentionally a single mutable operational artifact. It is
-- never a catalog and must not be used as a history store.
CREATE TABLE current_run (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    reference_time TEXT NOT NULL, forecast_end_exclusive TEXT NOT NULL,
    timestep_hours INTEGER NOT NULL CHECK (timestep_hours > 0),
    mgb_settings_json TEXT NOT NULL, spatial_settings_json TEXT NOT NULL,
    interpolation_settings_json TEXT NOT NULL, review_window_json TEXT NOT NULL,
    observed_providers_json TEXT NOT NULL,
    precipitation_qc_settings_json TEXT NOT NULL,
    saved_run_id TEXT, saved_run_description TEXT,
    responsible_person TEXT, reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE forecast_scenario (
    scenario_id TEXT PRIMARY KEY, position INTEGER NOT NULL UNIQUE CHECK (position >= 0),
    scenario_kind TEXT NOT NULL CHECK (scenario_kind IN ('zero', 'raw', 'corrected')),
    label TEXT NOT NULL, provider_code TEXT, source_asset_id TEXT, source_asset_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE forecast_correction (
    correction_id INTEGER PRIMARY KEY,
    scenario_id TEXT NOT NULL REFERENCES forecast_scenario(scenario_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0), t0_step INTEGER NOT NULL, t1_step INTEGER NOT NULL,
    shift_lat REAL NOT NULL DEFAULT 0, shift_lon REAL NOT NULL DEFAULT 0,
    rotation_deg REAL NOT NULL DEFAULT 0,
    multiplication_factor REAL NOT NULL DEFAULT 1 CHECK (multiplication_factor > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}', CHECK (t1_step >= t0_step), UNIQUE (scenario_id, position)
);
CREATE TABLE observed_replacement (
    station_id TEXT NOT NULL, observed_at TEXT NOT NULL, value REAL,
    PRIMARY KEY (station_id, observed_at)
);
CREATE TABLE station_exclusion (
    station_id TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
    PRIMARY KEY (station_id, start_time, end_time), CHECK (end_time > start_time)
);
CREATE TABLE current_execution (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    status TEXT NOT NULL CHECK (status IN ('never', 'running', 'completed', 'failed')),
    execution_id TEXT, started_at TEXT, finished_at TEXT, error_message TEXT
);
CREATE TABLE published_cache_reference (
    scenario_id TEXT PRIMARY KEY REFERENCES forecast_scenario(scenario_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL, published_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_forecast_correction_scenario ON forecast_correction(scenario_id, position);
CREATE INDEX idx_observed_replacement_time ON observed_replacement(observed_at);
CREATE INDEX idx_station_exclusion_window ON station_exclusion(station_id, start_time, end_time);
