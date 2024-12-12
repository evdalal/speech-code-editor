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

from flask import Blueprint, request, jsonify
from typing import Dict, Any
from flask.wrappers import Response
from app.ollama_utils import query_ollama_prompt, convert_model_output_to_json
from app.firebase_utils import update_user_messages_to_firebase, get_conversation_messages, update_string_data_to_firebase


app_views: Blueprint = Blueprint('app_views', __name__)
ASSISTANT_ROLE: str = "assistant"
USER_ROLE: str = "user"
OLLAMA_API: str = 'http://172.17.0.1:11434/api/chat'

@app_views.route('/context', methods=['POST'])
def context() -> Response:
    '''
    This is the primary method of the context endpoint. This endpoint updates the
    firebase database with a given file provided by the frontend. This file should
    be in JSON format with line numbers as keys and the contents of each line as
    the respective values. The userid and conversationid should be included as
    headers as well. 

    Required Headers:
    :str userid: The unique user ID for the given user.
    :str conversationid: The unique ID associated with the given message history.
    :JSON context: A JSON of the following format:
    {
        "1": "<Contents of line 1>",
        "2": "<Contents of line 2>",
        "3": "<Contents of line 3>",
        ...
    }

    Return:
    Status code and response message
    '''
    data: Dict = request.json

    # Guards to ensure properly formed JSON
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    if 'context' not in data:
        return jsonify({'error': 'Missing "context" key in JSON'}), 400
    if not isinstance(data['context'], dict):
        return jsonify({'error': '"context" should be a dictionary'}), 400
    try:
        userid: str = data['userid']
        convoid: str = data['conversationid']
        context: Dict = data['context']  # expecting 'context' to be a dictionary
    except KeyError:
        return jsonify({'error': 'Missing key in JSON. Keys must include context, userid, and conversationid'}), 400

    # Send to ContextGenerator to update context in Firebase
    update_user_messages_to_firebase(convoid, userid, context, USER_ROLE)

    return jsonify({'message': 'Context Received'})


@app_views.route('/prompt', methods=['POST'])
def prompt() -> Response:
    """
    This is the primary method for the prompt endpoint. It retrieves a user's
    message history, queries Ollama for code generation, and updates the database.

    Arguments:
    :str userid: The unique user ID for the given user.
    :str conversationid: The unique ID associated with the given message history.
    :str prompt: The natural language user prompt for new code.
    """

    # Retrieve JSON data from the POST request
    data: Dict = request.json
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    try:
        userid: str = data['userid']
        convoid: str = data['conversationid']
        prompt: str = data['prompt']
    except KeyError:
        return jsonify({'error': 'Missing key in JSON. Keys must include prompt, userid, and conversationid'}), 400

    # Gets prior context for the conversation
    messages_list: list = get_conversation_messages(convoid)

    # Update Firebase with the new user prompt message
    update_string_data_to_firebase(convoid, userid, prompt, USER_ROLE)
    
    new_prompt_dict: Dict[str, Any] = {
        "role": USER_ROLE,
        "content": prompt
    }

    # Append the current user prompt as a new dictionary entry in messages_list
    messages_list.append(new_prompt_dict)

    try:
        # Send structured messages to Ollama
        response_content: str = query_ollama_prompt(OLLAMA_API, messages_list)
        # Parse results
        json_response: Dict[str, Any] = convert_model_output_to_json(response_content)
    except Exception as e:
        return jsonify({'error': f'Ollama API request failed: {str(e)}'}), 500
    
    # Update Firebase with the LLM response
    try:
        update_string_data_to_firebase(convoid, userid, response_content, ASSISTANT_ROLE)
    except Exception as e:
        return jsonify({'error': f'Error updating Firebase: {str(e)}'}), 500
    if isinstance(json_response, str):
        return json_response
    response_data = {
        'success': True,
        'data': {
            'code': json_response.get('code', {}),
            'explanation': json_response.get('explanation', ''),
            'type': json_response.get('type', ''),
            'modify_range': json_response.get('modify-range'),
        }
    }
    return jsonify(response_data)