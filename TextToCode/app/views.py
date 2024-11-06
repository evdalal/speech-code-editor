from flask import Blueprint, request, jsonify
from typing import Any, Dict
from flask.wrappers import Response
app_views = Blueprint('app_views', __name__)

@app_views.route('/context', methods=['POST'])
def context() -> Response:
    data: Dict = request.json
    # Add guards to ensure properly formed JSON

    # Send to ContextGenerator to update context in Firebase

    # Return message
    return jsonify({'message': f'Context Received'})

@app_views.route('/prompt', methods=['POST'])
def prompt() -> Response:
    data: Dict = request.json
    # Add guards to ensure properly formed JSON

    # Send to PromptGenerator to query Firebase and send to Ollama

    # Return message
    return jsonify({'message': 'Prompt processed successfully', 'response': 'example response'})
