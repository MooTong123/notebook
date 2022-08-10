create table model_registry
(
    id INTERGER PRIMARY KEY ASC,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    registered_date TEXT DEFAULT CURRENT_TIMESTAMP NOT NULL,
    remote_path TEXT NOT NULL,
    stage TEXT DEFAULT 'DEVELOPMENT' NOT NULL
);
