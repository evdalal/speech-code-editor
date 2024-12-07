from typing import Dict, Any, List
from flask.wrappers import Response
import requests
import json
from app.exceptions import OllamaConnectionError, OllamaModelNotFoundError, OllamaResourceNotFoundError
from requests.exceptions import ConnectionError



def query_ollama_models(url: str):
    '''
    Queries a list of models available on Ollama
    '''


def query_ollama_prompt(url: str, message_list: list, model: str = 'llama3.2') -> dict:
    '''
    Queries Ollama given a list of messages and a model
    '''
    payload = {
        "model": model,
        "messages": message_list,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
    except ConnectionError as e:
        print(e)
        raise Exception('ERROR: Ollama server failed to connect.')

    if response.status_code == 200:
        try:
            # message_content = response.json().get('message')
            return response.json()  # parsing to json format
        except ValueError:
            # Handle nested JSON or malformed responses
            raw_text = response.text
            print("Malformed JSON response. Raw content:", raw_text)
            try:
                return json.loads(raw_text)  # Secondary parsing
            except json.JSONDecodeError:
                raise Exception("Unable to parse response content as JSON.")
    elif response.status_code == 404:
        error_message = response.json().get('error', 'Resource not found')
        if 'not found' in error_message:
            raise Exception(f'Model {model} could not be found.')
        raise Exception('Resource not found.')
    else:
        print(f"Request failed with status code {response.status_code}")
        return {"error": f"{response.status_code} - {response.text}"}