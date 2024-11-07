from typing import Dict
import pytest
from app.exceptions import OllamaConnectionError, OllamaModelNotFoundError, OllamaResourceNotFoundError
from app.ollama_utils import query_ollama_prompt

# example_out = {'model': 'llama3.2', 
#                'created_at': '2024-11-06T17:48:47.8270616Z', 
#                'message': {'role': 'assistant', 
#                            'content': '{\n    "code": {\n        "21": "def add_numbers(a, b):",\n        "22": "    return a + b",\n        "23": "",\n        "24": "# Example of using the add_numbers function",\n        "25": "result = add_numbers(5, 7)",\n        "26": "print(result)"\n    },\n    "explanation": "This code defines a function `add_numbers` that takes two numbers as input and returns their sum. It also includes an example usage of this function.",\n    "type": "ADD",\n    "modify-range": [None, None]\n}'}, 
#                            'done_reason': 'stop', 
#                            'done': True, 
#                            'total_duration': 10747391600, 
#                            'load_duration': 7407012100, 
#                            'prompt_eval_count': 729, 
#                            'prompt_eval_duration': 643058000, 
#                            'eval_count': 128, 
#                            'eval_duration': 2675105000           
#                 }

# TODO: update ollama json responses

def test_query_ollama_prompt_success(mock_requests_post):
    # Mock a successful response
    mock_requests_post(status_code=200, json_data={'message': 'Success'})

    # The following is not important because ollama is mocked
    url: str = 'http://127.0.0.1:11434/api/chat'
    messages: str = 'Hello, Ollama!'
    model: str = 'llama3.2'
    response: Dict = query_ollama_prompt(url, messages, model)
    
    assert response == {'message': 'Success'}


def test_query_ollama_prompt_connection_refused(mock_requests_post):
    # Mock a connection refused error
    mock_requests_post(status_code=400, raise_exception=ConnectionError)

    url = 'http://127.0.0.1:11434/api/chat'
    messages = 'Hello, Ollama!'
    model = 'llama3.2'
    
    with pytest.raises(OllamaConnectionError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'ERROR: Ollama server failed to connect.'


def test_query_ollama_prompt_connection_refused(mock_requests_post):
    # Mock a connection refused error
    mock_requests_post(status_code=400, raise_exception=ConnectionError)

    url = 'http://127.0.0.1:11434/api/chat'
    messages = 'Hello, Ollama!'
    model = 'llama3.2'
    
    with pytest.raises(OllamaConnectionError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'ERROR: Ollama server failed to connect.'


def test_query_ollama_prompt_model_not_found(mock_requests_post):
    # Mock a connection refused error
    mock_requests_post(status_code=404, json_data={'error': 'model "lama3.2" not found, try pulling it first'})

    url = 'http://127.0.0.1:11434/api/chat'
    messages = 'Hello, Ollama!'
    model = 'lama3.2'
    
    with pytest.raises(OllamaModelNotFoundError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'The model lama3.2 could not be found.'


def test_query_ollama_prompt_resource_not_found(mock_requests_post):
    # Mock a connection refused error
    mock_requests_post(status_code=404, json_data={'error': 'Some ollama resource cannot be found'})

    url = 'http://127.0.0.1:11434/api/chat'
    messages = 'Hello, Ollama!'
    model = 'lama3.2'
    
    with pytest.raises(OllamaResourceNotFoundError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'The requested resource could not be found.'