from typing import Dict
from flask.wrappers import Response
from flask import current_app
import requests
from app.exceptions import OllamaConnectionError, OllamaModelNotFoundError, OllamaResourceNotFoundError
import json
import re

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
    


def convert_to_json(input_string: str) -> dict:
    """
    Converts a string representation of a JSON-like object into a proper JSON object.
    Ensures single quotes are replaced with escaped double quotes and handles f-strings properly.

    Args:
        input_string (str): The input string to be converted.

    Returns:
        dict: A dictionary representation of the JSON.
    """
    try:
        # Step 1: Replace escaped newlines with actual newlines
        processed_string = input_string.replace('\\n', '\n')
        processed_string = re.sub(r"(?<!\\)'", '"', processed_string)
        processed_string = processed_string.replace("\\'", "'")

        # Step 3: Attempt to parse as JSON
        return json.loads(processed_string)


    except json.JSONDecodeError as e:

        current_app.logger.error(f"JSON Decode Error: {str(e)}")
        try:
            # Step 4: Remove everything before the first "{" and after the last "}"
            start_index = input_string.find('{')
            end_index = input_string.rfind('}')
            if start_index != -1 and end_index != -1:
                trimmed_string = input_string[start_index:end_index + 1]
                current_app.logger.info(f"Trimmed string for retry: {trimmed_string}")
                # Retry parsing the trimmed string
                return json.loads(trimmed_string)
            else:
                current_app.logger.error("No valid JSON structure found in the input string.")
                return input_string
        except json.JSONDecodeError as inner_e:
            current_app.logger.error(f"Retry JSON Decode Error: {str(inner_e)}")
            return input_string