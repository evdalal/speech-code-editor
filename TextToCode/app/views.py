from flask import Blueprint, request, jsonify
from typing import Any, Dict
from flask.wrappers import Response
import requests
# from app.ollama_utils import query_ollama_prompt
from app.firebase_utils import get_user_messages_from_firebase
from app.ollama_utils import query_ollama_prompt

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
    



    print(llm_message)



    # Return message
    return jsonify({'message': 'Prompt processed successfully', 'response': llm_message})
