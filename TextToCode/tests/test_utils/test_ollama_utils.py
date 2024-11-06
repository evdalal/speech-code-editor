# from unittest.mock import patch


# example_out = {'model': 'llama3.2', 
#                'created_at': '2024-11-06T17:48:47.8270616Z', 
#                'message': {'role': 'assistant', 
#                            'content': '{\n    "code": {\n        "21": "def add_numbers(a, b):",\n        "22": "    return a + b",\n        "23": "",\n        "24": "# Example of using the add_numbers function",\n        "25": "result = add_numbers(5, 7)",\n        "26": "print(result)"\n    },\n    "explanation": "This code defines a function `add_numbers` that takes two numbers as input and returns their sum. It also includes an example usage of this function.",\n    "type": "ADD",\n    "modify-range": [None, None]\n}'}, 
#                            'done_reason': 'stop', 
#                            'done': True, 
#                            'total_duration': 10747391600, 
#                            'load_duration': 7407012100, 
#                            'prompt_eval_count': 729, 
#                            'prompt_eval_duration': 643058000, 
#                            'eval_count': 128, 
#                            'eval_duration': 2675105000           
#                 }



# def test_ollama_prompt():
    
    
#     # Mock a successful response from Ollama API 
#     mock_post = mocker.patch('requests.post') 
#     mock_post.return_value.status_code = 200 
#     mock_post.return_value.json.return_value = {'result': 'test response'}

#     messages = ["Hello, Ollama!", "Can you generate a prompt for me?"]
#     model = 'llama3.2'

#     response = query_ollama_prompt(messages, model) 
#     # Ensure the POST request was called with the correct parameters 
#     mock_post.assert_called_once_with( 'https://api.ollama.ai/generate', json={'model': model, 'messages': messages} )
