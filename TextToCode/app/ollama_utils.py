from typing import Dict, Any, List
from flask.wrappers import Response
import requests
import json
from app.exceptions import OllamaConnectionError, OllamaModelNotFoundError, OllamaResourceNotFoundError




def query_ollama_models(url: str):
    '''
    Queries a list of models available on Ollama
    '''


def query_ollama_prompt(url: str, message_list: list, model: str = 'llama3.2-vision:11b') -> dict:
    '''
    Queries Ollama given a list of messages and a model
    '''
    payload: Dict = {
        "model": model,
        "messages": message_list,
        "stream": False
    }

    try:
        response: Response = requests.post(url, json=payload)
    except ConnectionError as e:
        print(e)
        raise OllamaConnectionError('ERROR: Ollama server failed to connect.')

    # TODO: Process ollama query and throw exceptions if we cant interpret data
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        # Error handling for when Ollama api cannot find entity or invalid entity is sent
        error_message: str = response.json().get('error', 'Resource not found')
        if 'not found, try pulling it first' in error_message:
            raise OllamaModelNotFoundError(f'The entity {model} could not be found.')
        else:  # catch-all for any other would-be 404 errors
            raise OllamaResourceNotFoundError('The requested resource could not be found.')
    else:
        print(f"Request failed with status code {response.status_code}")
        return f'Error: {response.status_code} - {response.text}'