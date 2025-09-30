-- FN Media AI Database Schema
-- Stores AI processing results and metadata

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum for processing status
CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed');

-- Photo analysis results table
CREATE TABLE IF NOT EXISTS photo_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL,
    photo_url TEXT NOT NULL,
    processing_status processing_status DEFAULT 'pending',
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- AI model results (stored as JSONB for flexibility)
    object_detection_results JSONB,
    scene_classification_results JSONB,
    ocr_results JSONB,
    openai_vision_results JSONB,

    -- Enhanced metadata
    generated_tags TEXT[],
    generated_description TEXT,
    location_inferences JSONB,
    brand_detections JSONB,

    -- Performance tracking
    processing_duration_ms INTEGER,
    model_versions JSONB,
    error_details JSONB
);

-- Object detections table (normalized for querying)
CREATE TABLE IF NOT EXISTS object_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    photo_analysis_id UUID NOT NULL REFERENCES photo_analyses(id) ON DELETE CASCADE,
    object_class TEXT NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    bounding_box JSONB NOT NULL, -- {x, y, width, height}
    model_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scene classifications table
CREATE TABLE IF NOT EXISTS scene_classifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    photo_analysis_id UUID NOT NULL REFERENCES photo_analyses(id) ON DELETE CASCADE,
    scene_class TEXT NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    model_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- OCR extractions table
CREATE TABLE IF NOT EXISTS ocr_extractions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    photo_analysis_id UUID NOT NULL REFERENCES photo_analyses(id) ON DELETE CASCADE,
    extracted_text TEXT NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    bounding_box JSONB, -- {x, y, width, height}
    language TEXT,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI processing metrics table
CREATE TABLE IF NOT EXISTS ai_processing_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    photo_analysis_id UUID REFERENCES photo_analyses(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_unit TEXT,
    model_name TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_photo_analyses_post_id ON photo_analyses(post_id);
CREATE INDEX IF NOT EXISTS idx_photo_analyses_status ON photo_analyses(processing_status);
CREATE INDEX IF NOT EXISTS idx_photo_analyses_confidence ON photo_analyses(confidence_score);
CREATE INDEX IF NOT EXISTS idx_photo_analyses_created_at ON photo_analyses(created_at);

CREATE INDEX IF NOT EXISTS idx_object_detections_analysis_id ON object_detections(photo_analysis_id);
CREATE INDEX IF NOT EXISTS idx_object_detections_class ON object_detections(object_class);
CREATE INDEX IF NOT EXISTS idx_object_detections_confidence ON object_detections(confidence_score);

CREATE INDEX IF NOT EXISTS idx_scene_classifications_analysis_id ON scene_classifications(photo_analysis_id);
CREATE INDEX IF NOT EXISTS idx_scene_classifications_class ON scene_classifications(scene_class);

CREATE INDEX IF NOT EXISTS idx_ocr_extractions_analysis_id ON ocr_extractions(photo_analysis_id);
CREATE INDEX IF NOT EXISTS idx_ocr_extractions_text ON ocr_extractions USING gin(to_tsvector('english', extracted_text));

CREATE INDEX IF NOT EXISTS idx_ai_metrics_analysis_id ON ai_processing_metrics(photo_analysis_id);
CREATE INDEX IF NOT EXISTS idx_ai_metrics_name ON ai_processing_metrics(metric_name);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_photo_analyses_updated_at
    BEFORE UPDATE ON photo_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE OR REPLACE VIEW photo_analyses_summary AS
SELECT
    pa.id,
    pa.post_id,
    pa.processing_status,
    pa.confidence_score,
    pa.created_at,
    pa.processing_duration_ms,
    COALESCE(array_length(pa.generated_tags, 1), 0) as tag_count,
    (SELECT COUNT(*) FROM object_detections od WHERE od.photo_analysis_id = pa.id) as object_count,
    (SELECT COUNT(*) FROM scene_classifications sc WHERE sc.photo_analysis_id = pa.id) as scene_count,
    (SELECT COUNT(*) FROM ocr_extractions oe WHERE oe.photo_analysis_id = pa.id) as ocr_count
FROM photo_analyses pa;

-- View for high-confidence results
CREATE OR REPLACE VIEW high_confidence_analyses AS
SELECT *
FROM photo_analyses
WHERE confidence_score >= 0.85
AND processing_status = 'completed';

-- Sample data for development (optional)
-- INSERT INTO photo_analyses (post_id, photo_url, processing_status, confidence_score, generated_tags, generated_description)
-- VALUES (
--     uuid_generate_v4(),
--     'https://example.com/sample-photo.jpg',
--     'completed',
--     0.92,
--     ARRAY['backpack', 'blue', 'outdoor'],
--     'A blue backpack found in a park setting'
-- );

-- Print schema creation completion
SELECT 'FN Media AI database schema created successfully' as status;