from flask import Blueprint, request, jsonify
from typing import Any, Dict
from flask.wrappers import Response
import requests
# from app.ollama_utils import query_ollama_prompt
from app.ollama_utils import query_ollama_prompt

from app.firebase_utils import get_user_messages_from_firebase
from app.firebase_utils import add_data_to_firebase


app_views = Blueprint('app_views', __name__)

OLLAMA_API = 'http://127.0.0.1:11434/api/chat'

@app_views.route('/context', methods=['POST'])
def context() -> Response:
    '''
    Sends file to the ContextCreator class, which 
    '''
    data: Dict = request.json

    # Guards to ensure properly formed JSON
    if data is None: 
        return jsonify({'error': 'Invalid JSON'}), 400 
    if 'context' not in data: 
        return jsonify({'error': 'Missing "context" key in JSON'}), 400
    if not isinstance(data['context'], str): 
        return jsonify({'error': '"context" should be a string'}), 400

    # Send to ContextGenerator to update context in Firebase

    # Return message
    return jsonify({'message': f'Context Received'})


@app_views.route('/prompt', methods=['POST'])
def prompt() -> Response:
    print('here')

    data: Dict = request.json

    # Guards to ensure properly formed JSON
    if data is None: 
        return jsonify({'error': 'Invalid JSON'}), 400 
    
    try:
        userid: str = data['userid']
        convoid: str = data['conversationid']
        prompt: str = data['prompt']
    except:
        return jsonify({'error': 'Missing key in JSON. Keys must include prompt, userid, and conversationid'}), 400

    # Get messages from firebase
    messages = get_user_messages_from_firebase(userid)
    prompt_message = {
        'role': 'user',
        'content': prompt,
        'format': 'json'
    }
    messages.append(prompt_message)

    # Send messages to Ollama
    ollama_response: Response = query_ollama_prompt(OLLAMA_API, messages)

    llm_message: Dict = ollama_response['message']

    # TODO: Add prompt and llm_message to firebase
    try:
        node = 'user-prompt'
        key = add_data_to_firebase(node, data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


    print(llm_message)



    # Return message
    return jsonify({'message': 'Prompt processed successfully', 'response': llm_message})

@app_views.route('/add_entry', methods=['POST'])
def add_entry() -> Any:
    """
    receive POST request and sent new data to Firebase
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        node = 'user-prompt'
        key = add_data_to_firebase(node, data)
        return jsonify({'success': True, 'key': key}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
