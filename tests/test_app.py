import pytest
from app import app as flask_app

@pytest.fixture
def client():
    with flask_app.test_client() as client:
        yield client

def test_home(client):
    """Test the home page loads successfully"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Image Matching System" in response.data

def test_upload_no_file(client):
    """Test upload endpoint without file"""
    response = client.post("/upload")
    assert response.status_code == 400
    assert b"No file uploaded" in response.data

def test_matched_image_not_found(client):
    """Test accessing non-existent matched image"""
    response = client.get("/matched/nonexistent.jpg")
    assert response.status_code == 404
