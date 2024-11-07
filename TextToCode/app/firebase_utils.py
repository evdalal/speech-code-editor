from typing import Dict, Any, List
import firebase_admin
from firebase_admin import credentials, db
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(base_dir, 'configuration/serviceAccountKey.json')
cred = credentials.Certificate(cred_path)
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://speech-to-code-26538-default-rtdb.firebaseio.com'
})


def get_user_messages_from_firebase(userid: str, conversation_id: str) -> List[Dict[str, Any]]:
    '''
    Queries Firebase Realtime Database and returns the list of user messages.

    :param userid (str): The user ID for which to query messages
    :return: List[Dict[str, Any]], the list of user messages
    '''

    mock = [
        {
        'role': 'system',
        'content': '''You are an expert programmer that helps to add or modify Python code 
        based on the user request, with concise explanations. Don't be too verbose. 
        Respond in JSON format to a user prompt with the headers code, explanation, type, and, modify-range.
        
        code: return the well-formed python code generated according to the user request.
        Do NOT reproduce the entire file. Only the requested portion by the user.
        The code should be in JSON format, nested inside the output json with each header as a line number.
        The line numbers should be determined by where the code should be placed in the file.
        For example, a method that should be added to line 21 should start with '21' as the first header of the code json.

        explanation: provide a brief explanation of what this code does and how it works.

        type: this should be 'ADD' if the generated code is to be added into the 
        file, and 'MODIFY' if the code is to replace existing code in the file.
        
        modify-range: If type is MODIFY, provide the line numbers to be changed in
        the original file, before modification. If type is ADD, leave null.
        
        Inputted context code is formatted in JSON format, with the header 
        corresponding to the current line number.

        Your output format should be the following:
        {
        'code': 
            {
                '1': '...',
                '2': '...',
                ...
            },
        'explanation': '...',
        'type': 'ADD or MODIFY'
        'modify-range': [<Start Line Num>, <End Line Num>]
        }

        Do NOT return anything other than JSON
        '''}
    ]
    try:
        # Reference the 'user-prompt' node
        ref = db.reference('user-prompt')
        # Retrieve all messages under the 'user-prompt' node
        messages = ref.get()

        # If no messages exist, return an empty list
        if not messages:
            return []

        # Filter messages where user_id and conversation_id match the parameters
        filtered_messages = [
            msg for msg in messages.values()
            if msg.get('user_id') == userid and msg.get('conversation_id') == conversation_id
        ]

        return filtered_messages

    except Exception as e:
        # Print an error message if any exception occurs
        print(f"Error retrieving messages for user {userid} with conversation ID {conversation_id}: {str(e)}")
        return []



def update_user_messages_to_firebase(node: str, data: dict):
    '''
    update the existing message or list of messages to the firebase storage for a user.
    '''
    pass

def add_data_to_firebase(node: str, data: dict) -> str:
    """
    Add data to a specified node in Firebase Realtime Database.
    :param node: str, the database node path
    :param data: dict, the data to be added
    :return: str, the generated key (unique ID)
    """
    ref = db.reference(node)
    new_entry = ref.push(data)
    return new_entry.key