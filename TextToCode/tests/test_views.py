import pytest
from flask.wrappers import Response
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_context(client):
    response: Response = client.post('/context', json={'context': 'example context'})
    assert response.status_code == 200

def test_context_missing_header(client):
    response: Response = client.post('/context', json={'contexto': 'file'})
    assert response.status_code == 400
    assert 'Missing "context" key in JSON' in response.get_json()['error']

def test_context_invalid_type(client):
    response: Response = client.post('/context', json={'context': 181})
    assert response.status_code == 400
    assert '"context" should be a string' in response.get_json()['error']


def test_prompt(client):
    example_prompt = 'Write a function which calculates the nth Fibonnaci value.'

    response: Response = client.post('/prompt', json={'prompt': example_prompt, 'userid': 1, 'conversationid': 1})
    
    
    
    assert response.status_code == 200
