# from unittest.mock import patch


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

import pytest
from app.ollama_utils import query_ollama_prompt

def test_query_ollama_prompt_success(mocker):
    # Mock a successful response
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'message': 'Success'}
    mocker.patch('app.ollama_utils.requests.post', return_value=mock_response)



    url = 'http://example.com'
    messages = 'Hello, Ollama!'
    model = 'llama3.2'
    response = query_ollama_prompt(url, messages, model)
    
    assert response == {'message': 'Success'}
