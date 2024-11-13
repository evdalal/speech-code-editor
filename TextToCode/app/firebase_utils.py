from typing import Dict, Any, List
import firebase_admin
from firebase_admin import credentials, db
import os
import time

test_prompt = """
You are an expert programmer that helps to add or modify Python code 
        based on the user request, with concise explanations. Don't be too verbose. 
        Respond in JSON format to a user prompt with the headers code, explanation, type, modify-range.
        
        code: return the well-formed python code generated according to the user request.
        Do NOT reproduce the entire file. Only the requested portion by the user.
        The code should be in JSON format, nested inside the output json with each header as a line number.
        The line numbers should be determined by where the code should be placed in the file.
        For example, a method that should be added to line 21 should start with '21' as the first header of the code json.
        Ensure that the code is indented, formatted, and placed correctly. The indentation should match that
        of the existing code.

        explanation: provide a brief explanation in plain text of what this code does and how it works.

        type: this should be 'ADD' if the generated code is to be added into the 
        file, and 'MODIFY' if the code is to replace existing code in the file.
        
        modify-range: If type is MODIFY, provide the minimal line numbers to be deleted in
        the original file and replaced with the new code for the current modification only. 
        If type is ADD, leave null.
        
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
"""


base_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(base_dir, 'configuration/serviceAccountKey.json')
cred = credentials.Certificate(cred_path)
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://speech-to-code-26538-default-rtdb.firebaseio.com'
})

def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    '''
        Queries Firebase Realtime Database and returns the list of user messages
        just based on conversation id

        :param conversation_id (str): The conversation ID to filter messages
        :return: List[Dict[str, Any]], the list of user messages
        '''
    try:
        # Directly reference the conversation_id as a key
        ref = db.reference(f'user-conversation/{conversation_id}')
        messages = ref.get()

        # If no messages exist, return an empty list
        if not messages:
            return []

        return []  # Return an empty list if user_id does not match

    except Exception as e:
        # Print an error message if any exception occurs
        print(f"Error retrieving messages with conversation ID {conversation_id}: {str(e)}")
        return []

def get_user_messages_from_firebase(userid: str, conversation_id: str) -> List[Dict[str, Any]]:
    '''
    Queries Firebase Realtime Database and returns the list of user messages.

    :param userid (str): The user ID for which to query messages
    :param conversation_id (str): The conversation ID to filter messages
    :return: List[Dict[str, Any]], the list of user messages
    '''
    try:
        # Directly reference the conversation_id as a key
        ref = db.reference(f'user-conversation/{conversation_id}')
        messages = ref.get()

        # If no messages exist, return an empty list
        if not messages:
            return []

        # Check if the user_id in the message matches the given userid
        if messages.get('user_id') == userid:
            return [messages]  # Return the message as a list

        return []  # Return an empty list if user_id does not match

    except Exception as e:
        # Print an error message if any exception occurs
        print(f"Error retrieving messages for user {userid} with conversation ID {conversation_id}: {str(e)}")
        return []

# TODO
def update_user_messages_to_firebase(conversation_id: str, user_id: str, data: dict, role: str):
    '''
    update the existing message or list of messages to the firebase storage for a user.
    '''
    # Define the path in the database for the conversation ID
    conversation_ref = db.reference(f'user-conversation/{conversation_id}')

    # Check if the conversation ID exists in the database
    if not conversation_ref.get():
        # If the conversation does not exist, initialize it with an empty dictionary
        conversation_ref.set({"history": {},
                              "user_id": user_id,
                              "system": test_prompt})
        print(f"Initialized new conversation for ID: {conversation_id}")


    # Generate a timestamp key to use as a unique identifier for the new message
    timestamp_key = str(int(time.time() * 1000))
    # Define the path in the database where the new message should be stored
    ref_path = f'user-conversation/{conversation_id}/history/{timestamp_key}'
    ref = db.reference(ref_path)

    # Define the data to be inserted, combining message and code data
    data = {
        "content": data,
        "role": role,
    }
    # Set the data at the specified location in Firebase
    ref.set(data)
    print(f"Message updated successfully for conversation {conversation_id} at {ref_path}")


def add_data_to_firebase(node: str, data: dict, custom_key: str = None) -> str:
    """
    Check whether  
    Add data to a specified node in Firebase Realtime Database with a custom key.
    :param node: str, the database node path
    :param data: dict, the data to be added
    :param custom_key: str, optional, the custom key name for the data entry
    :return: str, the key used for the new entry
    """
    ref = db.reference(node)

    if custom_key:
        # Use the provided custom key to set data at the specific path
        ref.child(custom_key).set(data)
        return custom_key
    else:
        # Use push to generate a unique key automatically
        new_entry = ref.push(data)
        return new_entry.key