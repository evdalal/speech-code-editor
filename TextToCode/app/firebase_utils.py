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

from typing import Dict, Any
from firebase_admin import credentials, db, initialize_app, App
from firebase_admin.db import Reference
import os
import time
import sys

with open('system_prompt.txt', 'r') as file:
    system_prompt: str = file.read()

base_dir: str = os.path.dirname(os.path.abspath(__file__))
cred_path: str = os.path.join(base_dir, 'configuration/serviceAccountKey.json')
cred: credentials.Certificate = credentials.Certificate(cred_path)
default_app: App = initialize_app(cred, {
    'databaseURL': 'https://speech-to-code-26538-default-rtdb.firebaseio.com'
})


def get_conversation_messages(conversation_id: str, limit:int=20) -> list:
    '''
    Queries Firebase Realtime Database and returns the list of user messages in 
    descending order by timestamp.

    :param conversation_id (str): The conversation ID to filter messages
    :return: List[Dict[str, Any]], the list of user messages in descending order
    '''
    try:
        # Reference the 'history' node under the specified conversation ID
        ref: Reference = db.reference(f'user-conversation/{conversation_id}/history')

        # Retrieve messages ordered by key (timestamp) and get the latest ones in reverse order.
        # Get 20 latest user conversation history block
        messages: Dict[str, Any] = ref.order_by_key().limit_to_last(limit).get()

        if not messages:
            return []
        messages_list: list = list(messages.values())
        messages_list.reverse()
        return messages_list

    except Exception as e:
        print(f"Error retrieving messages with conversation ID {conversation_id}: {str(e)}")
        return []


def dict_to_formatted_string(data: dict) -> str:
    """
    Helper function which converts a dictionary with line-number keys and 
    code-line values to a formatted long string, sorted by line numbers.

    :param data: Dictionary with line numbers as keys and code lines as values.
    :return: A formatted string representation of the dictionary, ordered by keys.
    """
    # Sort the dictionary by key, treating keys as integers to maintain numerical order
    sorted_items = sorted(data.items(), key=lambda item: int(item[0]))

    # Use a list to build the formatted string, which is more efficient
    lines = ["{\n"]
    lines.extend(f'    "{key}": "{value}",\n' for key, value in sorted_items)
    lines[-1] = lines[-1].rstrip(",\n") + "\n"  # Remove the last trailing comma and newline
    lines.append("}")

    return ''.join(lines)


def update_user_messages_to_firebase(conversation_id: str, user_id: str, data: dict, role: str):
    '''
    Update the database when the user message is in dictionary format.
    :param conversation_id (str): Unique conversation id associated with the
    user's current convo.
    :param user_id (str): Unique ID associated with current user.
    :param data (dict): Dictionary format of the user messages.
    :role (str): The role of the message (must be 'user', 'system' or 'assistant)
    '''
    formatted_string: str = dict_to_formatted_string(data)
    update_string_data_to_firebase(conversation_id, user_id, formatted_string, role)


def update_string_data_to_firebase(conversation_id: str, user_id: str, data: str, role: str):
    '''
    Update the database when the user message is in dictionary format.
    :param conversation_id (str): Unique conversation id associated with the
    user's current convo.
    :param user_id (str): Unique ID associated with current user.
    :param data (str): User messages already in formatted string.
    :role (str): The role of the message (must be 'user', 'system' or 'assistant)
    '''

    conversation_ref: Reference = db.reference(f'user-conversation/{conversation_id}')
    system_dict: Dict[str, Any] = {
        "role": 'system',
        'content': system_prompt
    }

    if not conversation_ref.get():
        # If the conversation does not exist, initialize it with an empty dictionary
        # make the system prompt displayed on the top of conversation history list
        conversation_ref.set({"history": {
            str(sys.maxsize): system_dict},
                              "user_id": user_id})
        print(f"Initialized new conversation for ID: {conversation_id}")

    # Generate a timestamp key to use as a unique identifier for the new message
    timestamp_key: str = str(int(time.time() * 1000))

    ref_path: str = f'user-conversation/{conversation_id}/history/{timestamp_key}'
    ref: Reference = db.reference(ref_path)
    history_data: Dict[str, Any] = {
        "content": data,
        "role": role,
    }
    # Set the data at the specified location in Firebase
    ref.set(history_data)
    print(f"Message updated successfully for conversation {conversation_id} at {ref_path}")