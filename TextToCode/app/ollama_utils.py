from typing import Dict
from flask.wrappers import Response
import requests
import json
from app.exceptions import OllamaConnectionError




def query_ollama_models(url: str):
    '''
    Queries a list of models available on Ollama
    '''


def query_ollama_prompt(url: str, messages: str, model: str = 'llama3.2') -> str:
    '''
    Queries Ollama given a list of messages and a model
    '''
    payload: Dict = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response: Response = requests.post(url, json=payload)
    except ConnectionError:
        raise OllamaConnectionError('ERROR: Ollama server failed to connect.')

    # TODO: Process ollama query and throw exceptions if we cant interpret data



    # Check if the request was successful 
    if response.status_code == 200: 
        return response.json()
    else: 
        return f'Error: {response.status_code} - {response.text}'