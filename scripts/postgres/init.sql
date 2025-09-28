-- PostgreSQL initialization script for FN Media AI
-- Sets up extensions and basic configurations

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create AI-specific database roles
CREATE ROLE fn_media_ai_readonly;
CREATE ROLE fn_media_ai_readwrite;

-- Grant permissions
GRANT CONNECT ON DATABASE findly_media_ai TO fn_media_ai_readonly;
GRANT CONNECT ON DATABASE findly_media_ai TO fn_media_ai_readwrite;

GRANT USAGE ON SCHEMA public TO fn_media_ai_readonly;
GRANT USAGE ON SCHEMA public TO fn_media_ai_readwrite;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO fn_media_ai_readonly;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fn_media_ai_readwrite;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO fn_media_ai_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fn_media_ai_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO fn_media_ai_readwrite;

-- Optimize for AI workloads
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET random_page_cost = 1.1;

-- Create AI processing status enum
CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed',
    'retrying'
);

-- Create confidence level enum
CREATE TYPE confidence_level AS ENUM (
    'very_low',
    'low',
    'medium',
    'high',
    'very_high'
);

-- Print completion message
SELECT 'PostgreSQL initialized for FN Media AI development' as status;