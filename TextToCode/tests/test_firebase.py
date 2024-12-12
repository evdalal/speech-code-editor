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

import os
from firebase_admin import credentials
from app.firebase_utils import update_user_messages_to_firebase, get_conversation_messages
import firebase_admin


# Firebase Initialization (only if not already initialized)
base_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(base_dir, 'configuration/serviceAccountKey.json')

# Check if Firebase has already been initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://speech-to-code-26538-default-rtdb.firebaseio.com'
    })

def test_update_and_get_user_messages():
    # Setup: Define test conversation ID, user ID, message, and file data
    conversation_id = "10086"
    role = "user"
    message1 = {"prompt": "Add a method in the class called transfer to the BankAccount class to transfer money from one account to another."}
    code = {
        1: "print('Hello World')",
        2: "print('Another line')"
    }
    user_id = "6X8yN3lRPPXqzTjrfURYFaJKsUW2"
    # Perform the update
    update_user_messages_to_firebase(conversation_id, user_id, message1, role)
    update_user_messages_to_firebase(conversation_id, user_id, code, role)


def test_get_conversation_messages():
    # Define the conversation_id to test with
    conversation_id = "10086"

    # Directly call the function to retrieve messages from Firebase
    result = get_conversation_messages(conversation_id)

    # Print the result to verify the function's output
    print(result)

    # Check that the function returns a list
    assert isinstance(result, list), "Expected result to be a list"

    # Check that there is a valid response (not None or an empty dictionary)
    assert result is not None, "Expected a valid result from Firebase, got None"


