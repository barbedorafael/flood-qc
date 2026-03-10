PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS provider (
    provider_code TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS variable (
    variable_code TEXT PRIMARY KEY,
    variable_name TEXT NOT NULL,
    default_unit TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS station (
    station_uid TEXT PRIMARY KEY,
    station_name TEXT NOT NULL,
    station_type TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'planned')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS station_alias (
    station_alias_id INTEGER PRIMARY KEY,
    station_uid TEXT NOT NULL REFERENCES station(station_uid) ON DELETE CASCADE,
    provider_code TEXT NOT NULL REFERENCES provider(provider_code),
    external_code TEXT NOT NULL,
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (provider_code, external_code),
    UNIQUE (station_uid, provider_code, external_code)
);

CREATE TABLE IF NOT EXISTS asset (
    asset_id TEXT PRIMARY KEY,
    asset_kind TEXT NOT NULL,
    format TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    provider_code TEXT REFERENCES provider(provider_code),
    checksum TEXT,
    valid_from TEXT,
    valid_to TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingest_batch (
    ingest_batch_id TEXT PRIMARY KEY,
    provider_code TEXT NOT NULL REFERENCES provider(provider_code),
    source_asset_id TEXT REFERENCES asset(asset_id),
    window_start TEXT,
    window_end TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'partial')),
    note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS observed_series (
    series_id TEXT PRIMARY KEY,
    station_uid TEXT NOT NULL REFERENCES station(station_uid) ON DELETE CASCADE,
    provider_code TEXT NOT NULL REFERENCES provider(provider_code),
    variable_code TEXT NOT NULL REFERENCES variable(variable_code),
    unit TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'raw' CHECK (state IN ('raw', 'curated', 'approved')),
    source_asset_id TEXT REFERENCES asset(asset_id),
    ingest_batch_id TEXT REFERENCES ingest_batch(ingest_batch_id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (station_uid, provider_code, variable_code, state, source_asset_id)
);

CREATE TABLE IF NOT EXISTS observed_value (
    series_id TEXT NOT NULL REFERENCES observed_series(series_id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (series_id, observed_at)
);

CREATE TABLE IF NOT EXISTS qc_flag (
    qc_flag_id INTEGER PRIMARY KEY,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    rule_code TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'accepted', 'rejected', 'resolved')),
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manual_edit (
    manual_edit_id INTEGER PRIMARY KEY,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    editor TEXT,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS run_catalog (
    run_id TEXT PRIMARY KEY,
    run_kind TEXT NOT NULL CHECK (run_kind IN ('automatic', 'manual')),
    parent_run_id TEXT,
    reference_time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'ready', 'executed', 'reviewed', 'published')),
    run_db_path TEXT NOT NULL,
    summary_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_station_alias_station_uid ON station_alias(station_uid);
CREATE INDEX IF NOT EXISTS idx_observed_series_station_var ON observed_series(station_uid, variable_code);
CREATE INDEX IF NOT EXISTS idx_observed_value_observed_at ON observed_value(observed_at);
CREATE INDEX IF NOT EXISTS idx_qc_flag_scope ON qc_flag(scope_type, scope_key);
CREATE INDEX IF NOT EXISTS idx_run_catalog_status ON run_catalog(status);

INSERT OR IGNORE INTO provider (provider_code, provider_name, provider_type) VALUES
    ('ana', 'Agencia Nacional de Aguas e Saneamento Basico', 'observed'),
    ('inmet', 'Instituto Nacional de Meteorologia', 'observed'),
    ('forecast_provider_x', 'Provider placeholder de previsao em grade', 'forecast'),
    ('mgb_setup_ref', 'Referencia de setup espacial externo do MGB', 'reference');

INSERT OR IGNORE INTO variable (variable_code, variable_name, default_unit, description) VALUES
    ('rain', 'Precipitacao observada', 'mm', 'Valor observado no timestamp original'),
    ('level', 'Nivel observado', 'cm', 'Nivel hidrometrico observado'),
    ('flow', 'Vazao observada', 'm3/s', 'Vazao hidrometrica observada'),
    ('rain_accum', 'Precipitacao acumulada', 'mm', 'Acumulado de chuva em janela temporal');