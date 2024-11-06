from typing import Dict, Any, List

def get_user_messages_from_firebase(userid: str) -> List[Dict[str, Any]]:
    '''
    Queries Firebase Firestore and returns the list of user messages.

    :param userid (str): The user for which to query messages 
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

    return mock


def update_user_messages_to_firebase(userid: str, messages: List[Dict] | Dict):
    '''
    Adds a message or list of messages to the firebase storage for a user.
    '''
    pass