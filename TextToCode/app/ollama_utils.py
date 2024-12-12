"""
# Project Name: Speech to Code
# Author: STC Team
# Date: 12/12/2024
# Last Modified: 12/12/2024
# Version: 1.0

# Copyright (c) 2024 Brown University
# All rights reserved.

# This file is part of the STC project.
# Usage of this file is restricted to the terms specified in the
# accompanying LICENSE file.

"""

from typing import Any, Dict
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

    # Check if the request was successful
    if response.status_code == 200:
        response_content: Dict[str, Any] = response.json()['message']['content']
        return response_content
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
    

def convert_model_output_to_json(input_string: str) -> dict:
    """
    Custom function that converts a string representation of a JSON-like object 
    into a proper JSON object. Ensures single quotes are replaced with escaped 
    double quotes and handles f-strings properly. For use on Ollama output.

    Args:
        input_string (str): The input string to be converted.

    Returns:
        dict: A dictionary representation of the JSON.
    """
    
    def remove_inline_comments(match):
        '''
        Nested function which removes inline comments outside key-value pairs.
        '''
        key_value = match.group(0)
        # Strip inline comments after the key-value pair
        return re.sub(r'\s*#.*$', '', key_value)

    try:
        # Replace escaped newlines with actual newlines
        processed_string: str = input_string.replace('\\n', '\n')

        # Replace single quotes with double quotes, handling escaped single quotes
        processed_string: str = re.sub(r"(?<!\\)'", '"', processed_string)
        processed_string: str = processed_string.replace("\\'", "'")

        # Remove inline comments outside key-value pairs
        processed_string: str = re.sub(
            r'"[^"]*"\s*:\s*".*?",?#.*$',
            remove_inline_comments,
            processed_string,
            flags=re.MULTILINE
        )
        return json.loads(processed_string)

    except json.JSONDecodeError as e:
        current_app.logger.error(f"JSON Decode Error: {str(e)}")
        try:
            # Remove everything before the first "{" and after the last "}"
            start_index: int = input_string.find('{')
            end_index: int = input_string.rfind('}')
            if start_index != -1 and end_index != -1:
                trimmed_string: str = input_string[start_index:end_index + 1]
                current_app.logger.info(f"Trimmed string for retry: {trimmed_string}")
                # Retry parsing the trimmed string
                return json.loads(trimmed_string)
            else:
                current_app.logger.error("No valid JSON structure found in the input string.")
                return input_string
        except json.JSONDecodeError as inner_e:
            current_app.logger.error(f"Retry JSON Decode Error: {str(inner_e)}")
            return input_string
