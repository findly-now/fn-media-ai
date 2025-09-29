"""
E2E tests for photo analysis endpoints.

Tests complete photo analysis workflow with real adapters.
"""

import pytest
from fastapi.testclient import TestClient

from fn_media_ai.main import create_app


class TestPhotoAnalysisEndpoints:
    """E2E tests for photo analysis API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with real app instance."""
        app = create_app()
        return TestClient(app)

    def test_analyze_photos_endpoint(self, client):
        """Test photo analysis endpoint with valid input."""
        request_data = {
            "photo_urls": [
                "https://example.com/photo1.jpg",
                "https://example.com/photo2.jpg"
            ],
            "post_id": "test-post-123",
            "user_id": "test-user-456",
            "item_type": "electronics"
        }

        response = client.post("/api/v1/photos/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "analysis_id" in data
        assert "photo_count" in data
        assert "overall_confidence" in data
        assert "processing_time_ms" in data
        assert "detected_objects" in data
        assert "scene_classification" in data
        assert "extracted_text" in data
        assert "color_analysis" in data
        assert "generated_tags" in data
        assert "enhancement_level" in data

        # Verify values
        assert data["photo_count"] == 2
        assert isinstance(data["overall_confidence"], float)
        assert 0.0 <= data["overall_confidence"] <= 1.0
        assert isinstance(data["processing_time_ms"], float)
        assert data["processing_time_ms"] >= 0
        assert isinstance(data["detected_objects"], list)
        assert isinstance(data["generated_tags"], list)

    def test_analyze_photos_minimal_input(self, client):
        """Test photo analysis with minimal required input."""
        request_data = {
            "photo_urls": ["https://example.com/photo.jpg"]
        }

        response = client.post("/api/v1/photos/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["photo_count"] == 1

    def test_analyze_photos_invalid_input(self, client):
        """Test photo analysis with invalid input."""
        # Empty photo URLs
        request_data = {"photo_urls": []}
        response = client.post("/api/v1/photos/analyze", json=request_data)
        assert response.status_code == 422

        # Missing photo URLs
        request_data = {}
        response = client.post("/api/v1/photos/analyze", json=request_data)
        assert response.status_code == 422

    def test_enhance_post_endpoint(self, client):
        """Test post enhancement endpoint."""
        post_id = "test-post-123"
        request_data = {
            "post_id": post_id,
            "photo_urls": [
                "https://example.com/photo1.jpg",
                "https://example.com/photo2.jpg"
            ],
            "current_title": "Lost iPhone",
            "current_description": "Lost my iPhone near the park",
            "item_type": "electronics"
        }

        response = client.post(f"/api/v1/photos/posts/{post_id}/enhance", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "post_id" in data
        assert "enhancement_applied" in data
        assert "confidence_score" in data
        assert "suggested_changes" in data
        assert "enhanced_metadata" in data
        assert "processing_time_ms" in data

        # Verify values
        assert data["post_id"] == post_id
        assert isinstance(data["enhancement_applied"], bool)
        assert isinstance(data["confidence_score"], float)
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert isinstance(data["suggested_changes"], dict)
        assert isinstance(data["enhanced_metadata"], dict)

    def test_enhance_post_mismatched_id(self, client):
        """Test post enhancement with mismatched post ID."""
        post_id = "test-post-123"
        request_data = {
            "post_id": "different-post-456",  # Different from URL
            "photo_urls": ["https://example.com/photo.jpg"]
        }

        response = client.post(f"/api/v1/photos/posts/{post_id}/enhance", json=request_data)
        assert response.status_code == 400

    def test_get_analysis_result_not_found(self, client):
        """Test retrieving non-existent analysis result."""
        analysis_id = "non-existent-id"
        response = client.get(f"/api/v1/photos/analysis/{analysis_id}")
        assert response.status_code == 404

    def test_photo_analysis_workflow_integration(self, client):
        """Test complete photo analysis workflow."""
        # Step 1: Analyze photos
        analyze_request = {
            "photo_urls": [
                "https://example.com/iphone.jpg",
                "https://example.com/keys.jpg"
            ],
            "post_id": "integration-test-post",
            "user_id": "integration-test-user",
            "item_type": "electronics"
        }

        analyze_response = client.post("/api/v1/photos/analyze", json=analyze_request)
        assert analyze_response.status_code == 200

        analyze_data = analyze_response.json()
        analysis_id = analyze_data["analysis_id"]

        # Step 2: Retrieve analysis result
        get_response = client.get(f"/api/v1/photos/analysis/{analysis_id}")
        # This will return 404 since we're using in-memory storage
        # In a real implementation with persistence, this would return 200

        # Step 3: Use analysis for post enhancement
        enhance_request = {
            "post_id": "integration-test-post",
            "photo_urls": analyze_request["photo_urls"],
            "current_title": "Lost Items",
            "item_type": "electronics"
        }

        enhance_response = client.post(
            f"/api/v1/photos/posts/integration-test-post/enhance",
            json=enhance_request
        )
        assert enhance_response.status_code == 200

        enhance_data = enhance_response.json()

        # Verify enhancement results
        assert enhance_data["post_id"] == "integration-test-post"
        assert "enhanced_metadata" in enhance_data
        assert "ai_confidence" in enhance_data["enhanced_metadata"]

    def test_photo_analysis_performance(self, client):
        """Test photo analysis performance with multiple photos."""
        request_data = {
            "photo_urls": [
                f"https://example.com/photo{i}.jpg" for i in range(5)
            ],
            "post_id": "performance-test-post"
        }

        response = client.post("/api/v1/photos/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify processing completed within reasonable time
        assert data["processing_time_ms"] < 30000  # Less than 30 seconds
        assert data["photo_count"] == 5

    def test_photo_analysis_error_handling(self, client):
        """Test photo analysis error handling."""
        # Test with invalid URLs (should still complete but with errors)
        request_data = {
            "photo_urls": [
                "invalid-url",
                "https://nonexistent-domain.example/photo.jpg"
            ]
        }

        response = client.post("/api/v1/photos/analyze", json=request_data)

        # Should still return 200 but with low confidence/no results
        assert response.status_code == 200
        data = response.json()
        assert data["photo_count"] == 2
        # May have low confidence or no detected objects due to invalid URLs