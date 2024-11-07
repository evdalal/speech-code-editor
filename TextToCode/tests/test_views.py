import pytest
from flask.wrappers import Response
from app import create_app

# TODO: add comments

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


def test_prompt_success(mock_requests_post, client, mocker):
    example_prompt: str = 'Write a function which calculates the nth Fibonnaci value.'

    # Mock functions 
    mocker.patch('app.firebase_utils.get_user_messages_from_firebase', return_value=[ {'role': 'system', 'content': 'Hello!'} ])     
    mock_requests_post(status_code=200, json_data={'message': 'Success'})

    response: Response = client.post('/prompt', json={'prompt': example_prompt, 'userid': 1, 'conversationid': 1})
        
    assert response.status_code == 200
    assert response.get_json()['message'] == "Prompt processed successfully"


def test_prompt_missing_header(client):
    response: Response = client.post('/prompt', json={'prompt': 'here', 'conversationid': 1})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing key in JSON. Keys must include prompt, userid, and conversationid'

    response: Response = client.post('/prompt', json={'prompt': 'here', 'userid': 1})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing key in JSON. Keys must include prompt, userid, and conversationid'
    
    response: Response = client.post('/prompt', json={'userid': 1, 'conversationid': 1})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing key in JSON. Keys must include prompt, userid, and conversationid'