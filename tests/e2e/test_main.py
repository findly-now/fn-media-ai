"""
E2E tests for FN Media AI main application.

Tests complete workflows with actual adapters and dependencies.
"""

import pytest
from fastapi.testclient import TestClient

from fn_media_ai.main import create_app


class TestMainApplication:
    """E2E tests for the main FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client with real app instance."""
        app = create_app()
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "fn-media-ai"
        assert data["version"] == "0.1.0"
        assert "description" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "fn-media-ai"

    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

    def test_api_documentation_available(self, client):
        """Test that API documentation is available."""
        # Test OpenAPI schema endpoint
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert schema["info"]["title"] == "FN Media AI"
        assert schema["info"]["version"] == "0.1.0"