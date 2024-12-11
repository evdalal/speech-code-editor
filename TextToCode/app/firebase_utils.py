from typing import Dict, Any, List
import firebase_admin
from firebase_admin import credentials, db
import os
import time
import sys

system_prompt = """
You are an expert Python programmer that helps to add or modify Python code based on the user request. Respond only in a valid JSON format with the following keys:

1. code:
Return the Python code snippet generated according to the user's request.
The code must:
Be well-formed and correctly indented.
Use double quotes for strings in the code and key names, ensuring it complies with JSON standards.
Escape special characters appropriately, such as " as \" or \ as \\.
Each line of code should be represented as a key-value pair, where the key is the line number, starting from where the code should be placed in the file.
Be concise and contain only the necessary logic requested by the user.

2. explanation:
Provide a concise and clear explanation of what the code does and how it works.
The explanation must:
Be written in proper English.
Avoid technical jargon and focus on clarity for the user.

3. type:
Indicate the type of action:
"ADD": If the code should be added to the file.
"MODIFY": If the code should replace existing code in the file.

4. modify-range:
If "type": "MODIFY", provide an array [start_line, end_line] specifying the range of line numbers to be replaced in the original file.
If "type": "ADD", set this field to null.
Input Details:
The inputted context code will be provided in JSON format. Each header corresponds to the current line number.
If the file is empty, your output should insert code starting from line 1.

Output Format:
Your output must be a valid JSON object. Use the following format exactly:
{
    "code": {
        "1": "...",
        "2": "...",
        ...
    },
    "explanation": "...",
    "type": "ADD or MODIFY",
    "modify-range": [<Start Line Num>, <End Line Num>] or null
}
Additional Notes:
1. Output must be valid JSON: Double-check to ensure proper escaping of characters like ", ' and \.
2. Do not include extra text or explanations outside the JSON object.
3. Ensure correct formatting, spacing, and syntax in the generated Python code to maintain Python standards.
4. If the input context requires a modification, carefully indicate the exact line numbers to be replaced in the "modify-range" field.
5. All comments must be on a separate line, enclosed by a separate line number key.
"""


base_dir = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(base_dir, 'configuration/serviceAccountKey.json')
cred = credentials.Certificate(cred_path)
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://speech-to-code-26538-default-rtdb.firebaseio.com'
})

def get_conversation_messages(conversation_id: str) -> list:
    '''
    Queries Firebase Realtime Database and returns the list of user messages in descending order by timestamp.

    :param conversation_id (str): The conversation ID to filter messages
    :return: List[Dict[str, Any]], the list of user messages in descending order
    '''
    try:
        # Reference the 'history' node under the specified conversation ID
        ref = db.reference(f'user-conversation/{conversation_id}/history')

        # Retrieve messages ordered by key (timestamp) and get the latest ones in reverse order.
        # Get 20 latest user conversation history block
        messages = ref.order_by_key().limit_to_last(20).get()

        # If no messages exist, return an empty list
        if not messages:
            return []
        messages_list = list(messages.values())
        messages_list.reverse()
        # Convert messages to a list of values, reversing the order for descending timestamps
        return messages_list

    except Exception as e:
        # Print an error message if any exception occurs
        print(f"Error retrieving messages with conversation ID {conversation_id}: {str(e)}")
        return []


# backup in case need userid
def get_user_messages_from_firebase(userid: str, conversation_id: str) -> list[Any] | list[object]:
    '''
    Queries Firebase Realtime Database and returns the list of user messages.

    :param userid (str): The user ID for which to query messages
    :param conversation_id (str): The conversation ID to filter messages
    :return:  list[Any] | list[object], the list of user messages
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

# helper
def dict_to_formatted_string(data: dict) -> str:
    """
    Converts a dictionary with line-number keys and code-line values to a formatted long string, sorted by line numbers.

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

    # Join all parts into a single string
    return ''.join(lines)

# code file
# TODO
def update_user_messages_to_firebase(conversation_id: str, user_id: str, data: dict, role: str):
    '''
    update the existing message in dictionary format to database
    Convert the code file data, which is in dictionary format, into a single long string that includes each key and its corresponding value.
    '''
    formatted_string = dict_to_formatted_string(data)
    update_string_data_to_firebase(conversation_id, user_id, formatted_string, role)


# for prompt / string
# save the data in string format to firebase database
def update_string_data_to_firebase(conversation_id: str, user_id: str, data: str, role: str):
    '''
    update the existing message or list of messages to the firebase storage for a user.
    '''
    # Define the path in the database for the conversation ID
    conversation_ref = db.reference(f'user-conversation/{conversation_id}')
    system_dict = {
        "role": 'system',
        'content': system_prompt
    }
    # Check if the conversation ID exists in the database
    if not conversation_ref.get():
        # If the conversation does not exist, initialize it with an empty dictionary
        # make the system prompt displayed on the top of conversation history list
        conversation_ref.set({"history": {
            str(sys.maxsize): system_dict},
                              "user_id": user_id})
        # console log
        print(f"Initialized new conversation for ID: {conversation_id}")


    # Generate a timestamp key to use as a unique identifier for the new message
    timestamp_key = str(int(time.time() * 1000))
    # Define the path in the database where the new message should be stored
    ref_path = f'user-conversation/{conversation_id}/history/{timestamp_key}'
    ref = db.reference(ref_path)
    # Define the data to be inserted, combining message and code data
    history_data = {
        "content": data,
        "role": role,
    }
    # Set the data at the specified location in Firebase
    ref.set(history_data)
    print(f"Message updated successfully for conversation {conversation_id} at {ref_path}")

# delete
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