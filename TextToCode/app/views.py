from flask import Blueprint, request, jsonify, current_app
from typing import Dict
from flask.wrappers import Response
from app.ollama_utils import query_ollama_prompt, convert_to_json
from app.firebase_utils import update_user_messages_to_firebase, get_conversation_messages
from app.firebase_utils import update_string_data_to_firebase
import json
import re

app_views = Blueprint('app_views', __name__)
# Constant string values
ASSISTANT_ROLE: str = "assistant"
USER_ROLE: str = "user"
# host machine's ip address
OLLAMA_API: str = 'http://172.17.0.1:11434/api/chat'


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



# # testing endpoint
# @app_views.route('/get_user_messages', methods=['POST'])
# def get_user_messages() -> Any:
#     """
#     Receives POST request with user ID and conversation ID as parameters
#     in the request body and returns the filtered messages from Firebase.
#     """
#     data = request.json  # Get the JSON data from the request body

#     if not data:
#         return jsonify({'error': 'No data provided'}), 400

#     userid = data.get('userid')
#     conversation_id = data.get('conversation_id')

#     if not userid or not conversation_id:
#         return jsonify({'error': 'Missing userid or conversation_id parameter'}), 400

#     try:
#         messages = get_user_messages_from_firebase(userid, conversation_id)
#         if not messages:
#             return jsonify({'message': 'No messages found'}), 404
#         return jsonify({'success': True, 'messages': messages}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
