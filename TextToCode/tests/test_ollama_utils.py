from typing import Dict
import pytest
from app.exceptions import OllamaConnectionError, OllamaModelNotFoundError, OllamaResourceNotFoundError
from app.ollama_utils import query_ollama_prompt

# example_out = {'model': 'llama3.2', 'created_at': '2024-11-13T01:20:15.0556498Z', 'message': {'role': 'assistant', 'content': '{\n  "code": {\n    "21": "    def transfer(self, amount, account_to_transfer_from):",\n    "22": "        if 0 < amount <= self.balance:",\n    "23": "            if account_to_transfer_from.balance >= amount:",\n    "24": "                account_to_transfer_from.balance -= amount",\n    "25": "                self.balance += amount",\n    "26": "                print(f\'${amount} transferred from ${account_to_transfer_from.account_number} to ${self.account_number}. New balances are ${account_to_transfer_from.balance} and ${self.balance}\')",\n    "27": "            else:",\n    "28": "                print(\'Insufficient funds in the source account.\')"\n    },\n  "explanation": "This method allows users to transfer money from one BankAccount object to another.",\n  "type": "ADD",\n  "modify-range": [null, null]\n}'}, 'done_reason': 'stop', 'done': True, 'total_duration': 6120845900, 'load_duration': 70446800, 'prompt_eval_count': 783, 'prompt_eval_duration': 1691466000, 'eval_count': 190, 'eval_duration': 4336498000}


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

    url: str = 'http://127.0.0.1:11434/api/chat'
    messages: str = 'Hello, Ollama!'
    model: str = 'llama3.2'
    
    with pytest.raises(OllamaConnectionError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'ERROR: Ollama server failed to connect.'


def test_query_ollama_prompt_model_not_found(mock_requests_post):
    # Mock a connection refused error
    mock_requests_post(status_code=404, json_data={'error': 'model "lama3.2" not found, try pulling it first'})

    url: str = 'http://127.0.0.1:11434/api/chat'
    messages: str = 'Hello, Ollama!'
    model: str = 'lama3.2'
    
    with pytest.raises(OllamaModelNotFoundError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'The model lama3.2 could not be found.'


def test_query_ollama_prompt_resource_not_found(mock_requests_post):
    # Mock a connection refused error
    mock_requests_post(status_code=404, json_data={'error': 'Some ollama resource cannot be found'})

    url: str = 'http://127.0.0.1:11434/api/chat'
    messages: str = 'Hello, Ollama!'
    model: str = 'lama3.2'
    
    with pytest.raises(OllamaResourceNotFoundError) as excinfo:
        query_ollama_prompt(url, messages, model)
    
    assert str(excinfo.value) == 'The requested resource could not be found.'