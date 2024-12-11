from flask import Blueprint, request, jsonify, current_app
from typing import Any, Dict
from flask.wrappers import Response
import requests
from app.ollama_utils import query_ollama_prompt
from app.firebase_utils import get_user_messages_from_firebase, update_user_messages_to_firebase, \
    get_conversation_messages
from app.firebase_utils import update_string_data_to_firebase
from app.firebase_utils import add_data_to_firebase
import json
import logging
import ast
import re

app_views = Blueprint('app_views', __name__)
# Constant string values
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"
# host machine's ip address
OLLAMA_API = 'http://172.17.0.1:11434/api/chat'


@app_views.route('/context', methods=['POST'])
def context() -> Response:
    '''
    Receives JSON data from the frontend, validates it, and sends it
    to the ContextCreator class for processing.
    '''
    data: Dict = request.json

    # Guards to ensure properly formed JSON
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400
    if 'context' not in data:
        return jsonify({'error': 'Missing "context" key in JSON'}), 400
    if not isinstance(data['context'], dict):
        return jsonify({'error': '"context" should be a dictionary'}), 400

    # Validate other necessary keys
    try:
        userid: str = data['userid']
        convoid: str = data['conversationid']
        context: Dict = data['context']  # expecting 'context' to be a dictionary
    except KeyError:
        return jsonify({'error': 'Missing key in JSON. Keys must include context, userid, and conversationid'}), 400

    # Send to ContextGenerator to update context in Firebase
    update_user_messages_to_firebase(convoid, userid, context, USER_ROLE)

    # Return message
    return jsonify({'message': 'Context Received'})


@app_views.route('/prompt', methods=['POST'])
def prompt() -> Response:
    """
    Endpoint to handle a user prompt request. This function:
    1. processes the prompt
    2. fetches conversation context from Firebase,
    3. sends the prompt and context to Ollama for response
    4. stores the LLM response back to Firebase.

    :return: JSON response containing a success message and the LLM response.
    """

    # Retrieve JSON data from the POST request
    data: Dict = request.json

    # Guards to ensure properly formed JSON
    if data is None:
        return jsonify({'error': 'Invalid JSON'}), 400

    # Extract required fields from the JSON request data
    try:
        userid: str = data['userid']
        convoid: str = data['conversationid']
        prompt: str = data['prompt']
    except KeyError:
        return jsonify({'error': 'Missing key in JSON. Keys must include prompt, userid, and conversationid'}), 400

    # Gets prior context for the conversation
    messages_list = get_conversation_messages(convoid)

    # Update Firebase with the new user prompt message
    update_string_data_to_firebase(convoid, userid, prompt, USER_ROLE)
    new_prompt_dict = {
        "role": USER_ROLE,
        "content": prompt
    }

    # Append the current user prompt as a new dictionary entry in messages_list
    messages_list.append(new_prompt_dict)

    # Send structured messages to Ollama
    try:
        ollama_response = query_ollama_prompt(OLLAMA_API, messages_list)
    except Exception as e:
        return jsonify({'error': f'Ollama API request failed: {str(e)}'}), 500
    response_content = ollama_response['message']['content']
    # transfer the response content to json format
    json_response = convert_to_json(response_content)
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
    # Debug: Log the type and content of json_response
    # current_app.logger.debug(f"json_response type: {type(json_response)}, content: {json_response}")
    return jsonify(response_data)


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


# testing endpoint
@app_views.route('/get_user_messages', methods=['POST'])
def get_user_messages() -> Any:
    """
    Receives POST request with user ID and conversation ID as parameters
    in the request body and returns the filtered messages from Firebase.
    """
    data = request.json  # Get the JSON data from the request body

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    userid = data.get('userid')
    conversation_id = data.get('conversation_id')

    if not userid or not conversation_id:
        return jsonify({'error': 'Missing userid or conversation_id parameter'}), 400

    try:
        messages = get_user_messages_from_firebase(userid, conversation_id)
        if not messages:
            return jsonify({'message': 'No messages found'}), 404
        return jsonify({'success': True, 'messages': messages}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
