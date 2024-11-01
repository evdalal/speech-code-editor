import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_context(client):
    response = client.post('/context', json={'context': 'example context'})
    assert response.status_code == 200

def test_prompt(client):
    response = client.post('/prompt', json={'prompt': 'example prompt'})
    assert response.status_code == 200
