# speech-code-editor

## Installation

- ### Set Up the Environments for Backend Applications
#### Step 1: Install NVIDIA GPU Driver
Download and install the appropriate NVIDIA GPU driver for your system:
- [NVIDIA GPU Driver Download](https://www.nvidia.com/en-us/drivers/)

#### Step 2: Install NVIDIA Container Toolkit
Follow the official guide to install the NVIDIA Container Toolkit, which is required to enable GPU support in Docker:
- [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)

#### Step 3: Install Docker
Install Docker to manage and run containerized applications:
- [Docker Installation Manual](https://docs.docker.com/desktop/?_gl=1*1d2byo4*_gcl_au*OTc4Nzg2NTA0LjE3Mjg1NzQxNjk.*_ga*MjAzMzE0NzQ5Mi4xNzI4NTc0MTcw*_ga_XJWPQMJYHQ*MTczNDA0MDk4OC4yNy4xLjE3MzQwNDA5ODkuNTkuMC4w)

#### Step 4: Install Docker Compose
Install Docker Compose to manage multi-container Docker applications:
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Notes:
1. **Verify Installations**:
   - Run `nvidia-smi` to ensure the NVIDIA driver and GPU are properly installed.
   - Run `docker --version` to confirm Docker is installed.
   - Run `docker compose version` to verify Docker Compose installation.

2. **Optional**: Configure Docker to start automatically with the system.
   ```bash
   sudo systemctl enable docker
   sudo systemctl start docker

- ### Run Speech To Text Service
1. In Windows and MacOS: Open the **Docker Desktop** application to ensure Docker is running.
   In Linux: 
   
   ```bash
   sudo systemctl start docker
   ```

2. cd to the SpeechToText folder
   
   ```bash
   cd /path/to/SpeechToText
   ```

3. Build the Docker container and start the service:
   
   ```bash
   docker-compose up --build
   ```
   
   * Docker will automatically download all necessary dependencies and libraries.
   
   * This command builds and runs the Docker container.
     
     

4. To stop the service: Press Ctrl+C in the terminal or run:
   
   ```bash
   docker-compose down
   ```
- ### Run Text To Code Service
1. Download Ollama Image on device
   
   ```bash
   docker pull ollama/ollama
   ```

2. Start the container:
      CPU Only (very slow)
   
   ```bash
   docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   ```
   
      Running on Nvidia GPU
   
   ```bash
   docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   ```
   
      Running on AMD GPU
   
   ```bash
   docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
   ```

3. Install Llama 3.2 via Ollama:
   
   ```bash
   docker exec -it ollama ollama pull llama3.2
   ```

4. cd to the TextToCode folder:
   
   ```bash
   cd /path/to/TextToCode
   ```
   
5. Build the Docker container and start the service:
   
   ```bash
   docker-compose up --build
   ```

## Usage

## Frontend

## Speech-To-Text

### API Specifications

### WebSocket Endpoint

**URL**: `ws://<server-ip>:<port-number>`

---

### Client to Server Communication

Clients send binary messages containing metadata and audio data.

**Message Format**:

- **First 4 bytes**: Metadata length (integer, little-endian).
- **Next `metadata_length` bytes**: JSON metadata as a UTF-8 string.
- **Remaining bytes**: Audio data (16-bit PCM format).

**Metadata Fields**:

| Field        | Type   | Description                            |
| ------------ | ------ | -------------------------------------- |
| `sampleRate` | Number | The sample rate of the incoming audio. |

---

### Server to Client Communication

The server sends JSON messages to deliver transcription updates.

**Message Types**:

1. **Real-Time Transcription**:
   
   ```
   {
       "type": "realtime",
       "text": "<transcribed_text>"
   }
   ```
   
   - Sent frequently with portions of text detected in real-time.

2. **Full Sentence Transcription**:
   
   ```
   {
       "type": "fullSentence",
       "text": "<complete_sentence>"
   }
   ```
   
   - Sent when a complete sentence has been transcribed.

---



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

### 1. get_conversation_messages

Retrieve the latest user messages from Firebase for a specific conversation ID.

**Parameters**

* `conversation_id` (str): The unique identifier for the conversation.
* `limit` (int, optional): The number of most recent messages to retrieve. Default is 20.

**Returns**

* `list`: A list of dictionaries representing user messages in descending order by timestamp.

###### **Example Usage**

```python
messages = get_conversation_messages("conversation123", limit=10) 
print(messages)
```

### **2. dict_to_formatted_string**

Convert a dictionary of line-number keys and code-line values into a formatted string sorted by line numbers.

**Parameters**

* `data` (dict): A dictionary with line numbers as keys and code lines as values.

**Returns**

* `str`: A formatted string representation of the dictionary, ordered by line numbers.

**Example Usage**

```python
data = {"2": "print('world')", "1": "print('hello')"}
formatted_string = dict_to_formatted_string(data)
print(formatted_string)
```



### **3. update_user_messages_to_firebase**

Update Firebase with user messages when the data is in dictionary format.

### **Parameters**

* `conversation_id` (str): The unique identifier for the conversation.
* `user_id` (str): The unique identifier for the user.
* `data` (dict): User messages in dictionary format.
* `role` (str): The role of the message (`'user'`, `'system'`, or `'assistant'`).

### **Example Usage**

```python
data = {"1": "Hello", "2": "How are you?"}
update_user_messages_to_firebase("conversation123", "user456", data, "user")
```



### **4. update_string_data_to_firebase**

Update Firebase with user messages when the data is already in a formatted string.

### **Parameters**

* `conversation_id` (str): The unique identifier for the conversation.
* `user_id` (str): The unique identifier for the user.
* `data` (str): User messages in a formatted string.
* `role` (str): The role of the message (`'user'`, `'system'`, or `'assistant'`).

### **Example Usage**

```python
formatted_data = "{\n    \"1\": \"Hello\",\n    \"2\": \"How are you?\"\n}"
update_string_data_to_firebase("conversation123", "user456", formatted_data, "user")
```
