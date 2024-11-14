import pytest
from flask import Flask, jsonify
from flask.wrappers import Response
from app import create_app
import json
# TODO: add comments

def test_context(client):
    response: Response = client.post('/context', json={'context': 'example context'})
    assert response.status_code == 200

def test_context_missing_header(client):
    response: Response = client.post('/context', json={'contexto': 'file'})
    assert response.status_code == 400
    assert 'Missing "context" key in JSON' in response.get_json()['error']

def test_context_invalid_type(client):
    response: Response = client.post('/context', json={'context': 181})
    assert response.status_code == 400
    assert '"context" should be a string' in response.get_json()['error']


def test_prompt_success(mock_requests_post, client, mocker):
    example_prompt: str = 'Write a function which calculates the nth Fibonnaci value.'

    # Mock functions 
    mocker.patch('app.firebase_utils.get_user_messages_from_firebase', return_value=[ {'role': 'system', 'content': 'Hello!'} ])     
    mock_requests_post(status_code=200, json_data={'message': 'Success'})

    response: Response = client.post('/prompt', json={'prompt': example_prompt, 'userid': 1, 'conversationid': 1})
        
    assert response.status_code == 200
    assert response.get_json()['message'] == "Prompt processed successfully"


def test_prompt_missing_header(client):
    response: Response = client.post('/prompt', json={'prompt': 'here', 'conversationid': 1})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing key in JSON. Keys must include prompt, userid, and conversationid'

    response: Response = client.post('/prompt', json={'prompt': 'here', 'userid': 1})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing key in JSON. Keys must include prompt, userid, and conversationid'
    
    response: Response = client.post('/prompt', json={'userid': 1, 'conversationid': 1})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing key in JSON. Keys must include prompt, userid, and conversationid'




# Test function to verify the /context endpoint with a valid payload
def test_context_endpoint(client):
    # Define the JSON payload to send to the /context endpoint
    test_data = {
        "context": {
            "1": "class BankAccount:",
            "2": "    def __init__(self, account_number, account_holder):",
            "3": "        self.account_number = account_number",
            "4": "        self.account_holder = account_holder",
            "5": "        self.balance = 0.0",
            "6": "",
            "7": "    def deposit(self, amount):",
            "8": "        if amount > 0:",
            "9": "            self.balance += amount",
            "10": "            print(f\"${amount} deposited. New balance is ${self.balance}\")",
            "11": "        else:",
            "12": "            print(\"Deposit amount must be positive.\")",
            "13": "",
            "14": "    def withdraw(self, amount):",
            "15": "        if 0 < amount <= self.balance:",
            "16": "            self.balance -= amount",
            "17": "            print(f\"${amount} withdrawn. New balance is ${self.balance}\")",
            "18": "        else:",
            "19": "            print(\"Insufficient funds or invalid withdrawal amount.\")",
            "20": "",
            "21": "    def get_balance(self):",
            "22": "        return self.balance",
            "23": "",
            "24": "# Example of creating a BankAccount object",
            "25": "account = BankAccount(\"123456789\", \"John Doe\")",
            "26": "account.deposit(1000)",
            "27": "account.withdraw(500)",
            "28": "print(account.get_balance())"
        },
        "userid": "6X8yN3lRPPXqzTjrfURYFaJKsUW2",
        "conversationid": "10086"
    }

    # Send a POST request to the /context route with the test data as JSON
    response: Response = client.post('/context', json=test_data)

    # Assert that the HTTP status code of the response is 200 (OK)
    assert response.status_code == 200
    # Assert that the response JSON matches the expected message
    assert response.get_json() == {'message': 'Context Received'}

# Test for missing 'context' key in the request
def test_context_missing_key(client):
    response: Response = client.post('/context', json={'contexto': 'file'})
    assert response.status_code == 400
    assert 'Missing "context" key in JSON' in response.get_json()['error']

# Test for invalid 'context' type (should be a dictionary)
def test_context_invalid_type(client):
    response: Response = client.post('/context', json={'context': 181})
    assert response.status_code == 400
    assert '"context" should be a dictionary' in response.get_json()['error']

# Test prompt processing with a mock response to check the /prompt endpoint
def test_prompt_success(client, mocker):
    example_prompt: str = 'Write a function which calculates the nth Fibonacci value.'

    # Mock dependent function call to get_user_messages_from_firebase
    mocker.patch('app.firebase_utils.get_user_messages_from_firebase', return_value=[{'role': 'system', 'content': 'Hello!'}])
    # Mock the external POST request and simulate a successful response
    mocker.patch('requests.post', return_value=mocker.Mock(status_code=200, json=lambda: {'message': 'Success'}))

    # Send the prompt data to /prompt route
    response: Response = client.post('/prompt', json={'prompt': example_prompt, 'userid': "1", 'conversationid': "1"})

    # Assert the HTTP status code and check the success message in the response
    assert response.status_code == 200
    assert response.get_json()['message'] == "Prompt processed successfully"


# Test function for the /prompt endpoint
def test_prompt_route(client):
    # Define test data for the POST request
    test_data = {
        'userid': '6X8yN3lRPPXqzTjrfURYFaJKsUW2',
        'conversationid': '10086',
        'prompt': 'Add a method in the class called transfer to the BankAccount class to transfer money from one account to another.'
    }

    # Send the POST request to the /prompt route
    response = client.post('/prompt', json=test_data)

    # Check that the function returned a response and print the LLM response content
    response_json = response.get_json()
    print("Ollama Response Message:", response_json.get('response'))

    # Assertions to check the response status and ensure content was processed
    assert response.status_code == 200, "Expected status code 200"
    assert response_json['message'] == 'Prompt processed successfully', "Expected success message"

