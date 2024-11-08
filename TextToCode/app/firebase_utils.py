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



def update_user_messages_to_firebase(node: str, data: dict):
    '''
    update the existing message or list of messages to the firebase storage for a user.
    '''
    pass

def add_data_to_firebase(node: str, data: dict, custom_key: str = None) -> str:
    """
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