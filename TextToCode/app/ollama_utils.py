from typing import Dict
from flask.wrappers import Response
import requests
import json




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
    response: Response = requests.post(url, json=payload)
    
    # must error handle ollama response

    # Check if the request was successful 
    if response.status_code == 200: 
        return response.json()
    else: 
        return f'Error: {response.status_code} - {response.text}'