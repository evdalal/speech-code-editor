from flask import Blueprint, request, jsonify
from typing import Any, Dict
from flask.wrappers import Response
import requests
from app.ollama_utils import query_ollama_prompt
from app.firebase_utils import get_user_messages_from_firebase, update_user_messages_to_firebase, get_conversation_messages
from app.firebase_utils import update_string_data_to_firebase
from app.firebase_utils import add_data_to_firebase
import json

app_views = Blueprint('app_views', __name__)
# Constant string values
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"

OLLAMA_API = 'http://127.0.0.1:11434/api/chat'


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

    print('here')

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

    # Update Firebase with the LLM response
    try:
        update_string_data_to_firebase(convoid, userid, ollama_response['message']['content'], ASSISTANT_ROLE)
    except Exception as e:
        return jsonify({'error': f'Error updating Firebase: {str(e)}'}), 500

    # Return success message and response
    return jsonify({'message': 'Prompt processed successfully', 'response': ollama_response})



# # testing endpoint
# @app_views.route('/add_entry', methods=['POST'])
# def add_entry() -> Any:
#     """
#     receive POST request and sent new data to Firebase
#     """
#     data = request.json
#     # Extract conversation_id from data
#     conversation_id = data.pop('conversation_id', None)
#
#     if not data:
#         return jsonify({'error': 'No data provided'}), 400
#
#     try:
#         node = 'user-conversation'
#         key = add_data_to_firebase(node, data, conversation_id)
#         return jsonify({'success': True, 'key': key}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


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
