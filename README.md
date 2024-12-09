# speech-code-editor

## Installation


## Usage


## Frontend


## Speech-To-Text


## Text-To-Code

### API Documentation

1. Endpoint: `/add-context`
    * Method: `POST`
    * Input: Context file
    * Response: The response will provide an HTTP status code and a message in the following JSON format:
    ```
    {
        'id': '<HTTP Status Code>',
        'message': '<Relevant Response Message>'
    }
    ```
2. Endpoint: `/prompt`
    * Method: `POST`
    * Input: JSON payload containing the user prompt
    * Notes: The payload should be in the following format:
    ```
    {
        'userid': '<User ID>',
        'convoid': '<Conversation ID>'
        'prompt': '<User Prompt>'
    }
    ```
    * Response: The response will provide the following JSON format:
    ```
    {
        'id': '<HTTP Status Code>',
        'code': {
            'n': '<Code for line n>',
            'n+1': '<Code for line n+1>',
            ...
        },
        'explanation': '<Brief explanation of the code generated by the model>',
        'type': '<ADD or MODIFY>'
        'modify_range': [Range of line numbers to replace with modification]
    }
    ```
    The model will generate code as well as the line numbers to place the code on.
    Based on user input, it will also determine if the code is to be added or modified.
    if `'type': 'ADD'`, the provided code should be added to the file starting at the
    line specified in the code JSON. If `'type': 'MODIFY'`, Use the `modify_range` header
    to determine what lines are replaced with the new code.

3. Endpoint: `/list-models`
    * Method: `GET`
    * Input: None
    * Response: JSON payload with list of supported models in the following format:
    ```
    {
        'id': '<HTTP Status Code>',
        'models': [
            'llama-3.2-3B',
            ...
        ]
    }
    ```


## Firebase

