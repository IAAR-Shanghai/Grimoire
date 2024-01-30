# LLM Vllm Deployment Example

## Environment Setup

```bash
pip install vllm
```

Note: If using GPU A800, it is recommended to use ray version 2.6.3.

## Model Download

First, download the required model from Hugging Face, for example, Qwen-7B-Chat. Place the model files in the specified path, such as /home/model/Qwen-7B-Chat.

## Start the Service

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server \
    --model /home/model/Qwen-7B-Chat \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code 
```

## Access the Model

Now you can access Qwen-7B-Chat via `http://localhost:8000/generate`:

**Access using bash:**

```bash
curl http://localhost:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "n": 1,
        "max_tokens": 16,
        "temperature": 0
    }'
```

Example response:

```json
{
    "text":["San Francisco is a city in California, United States. It is located on the west coast of the"]
}
```

**Access using Python script:**

```python
import requests
import json

url = "http://localhost:8000/generate"
payload = json.dumps({
    "prompt": 'San Francisco is a',
    "temperature": 0.1,
    "max_tokens": 16,
    "n": 1
})
headers = {
    'Content-Type': 'application/json'
}
res = requests.request("POST", url, headers=headers, data=payload)
res = res.json()['text'][0]  # VLLM will automatically append the query to the response, so here you can remove it.
return res
```

Example response:

```bash
>>> res
'San Francisco is a city in California, located on the west coast of the United States. It is'
```

# LLM FastApi Deployment Example

## Environment Setup

Create a new conda environment and install the following dependencies:

```bash
pip install fastapi
pip install uvicorn
pip install requests
pip install transformers
pip install streamlit
pip install sentencepiece
pip install accelerate
```

## Model Preparation

Download the required model from Hugging Face, for example, internlm2-chat-7b. Place the model files in the specified path, such as /home/model/internlm2-chat-7b.

## API Deployment Code

Create a file api.py in the same path as the model files (/home/model/) with the following code:

```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

# Set device parameters
DEVICE = "cuda"  # Use CUDA
DEVICE_ID = "0"  # CUDA device ID, empty if not set
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # Combine CUDA device information

# Function to clean GPU memory
def torch_gc():
    if torch.cuda.is_available():  # Check if CUDA is available
        with torch.cuda.device(CUDA_DEVICE):  # Specify CUDA device
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.ipc_collect()  # Collect CUDA memory fragments

# Create FastAPI application
app = FastAPI()

# Endpoint to handle POST requests
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # Declare global variables for using the model and tokenizer inside the function
    json_post_raw = await request.json()  # Get JSON data from POST request
    json_post = json.dumps(json_post_raw)  # Convert JSON data to a string
    json_post_list = json.loads(json_post)  # Convert the string to a Python object
    prompt = json_post_list.get('prompt')  # Get the prompt from the request
    history = json_post_list.get('history')  # Get the history from the request
    max_length = json_post_list.get('max_length')  # Get the maximum length from the request
    top_p = json_post_list.get('top_p')  # Get the top_p parameter from the request
    temperature = json_post_list.get('temperature')  # Get the temperature parameter from the request
    # Call the model for dialogue generation
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,  # Use 2048 as default if max_length is not provided
        top_p=top_p if top_p else 0.7,  # Use 0.7 as default if top_p is not provided
        temperature=temperature if temperature else 0.95  # Use 0.95 as default if temperature is not provided
    )
    now = datetime.datetime.now()  # Get the current time
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format the time as a string
    # Build the response JSON
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    # Build the log information
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # Print the log
    torch_gc()  # Execute GPU memory cleanup
    return answer  # Return the response

# Main function entry point
if __name__ == '__main__':
    # Load the pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("/home/model/internlm2-chat-7b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/model/internlm2-chat-7b", trust_remote_code=True).to(torch.bfloat16).cuda()
    model.eval()  # Set the model to evaluation mode
    # Start the FastAPI application
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)  # Start the application on the specified port and host
```

## Start the API Service

In the directory of the api.py file, run the command:

```bash
python api.py
```

Access using curl:

```bash
curl -X POST "http://localhost:8000"      
     -H 'Content-Type: application/json'      
     -d '{"prompt": "你好！", "history": []}'
```

Example response:

```json
{
    "response":"你好！很高兴为您服务。请问有什么我可以帮助您的吗？ ",
    "history":[["你好！","你好！很高兴为您服务。请问有什么我可以帮助您的吗？ "]],
    "status":200,"time":"2024-01-22 17:18:33"
}
```

Access using Python's request library:

```python
import requests
import json

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt, "history": []}
    response = requests.post(url='http://localhost:8000', headers=headers, data=json.dumps(data))
    return response.json()

if __name__ == '__main__':
    print(get_completion('你好！'))
```

Example response:

```json
{
    'response': '你好！很高兴为您服务，有什么需要帮助的吗？ ', 
    'history': [['你好！', '你好！很高兴为您服务，有什么需要帮助的吗？ ']], 
    'status': 200, 'time': '2024-01-22 17:20:43'
}
```

## More Information

- [FastApi_demo](https://github.com/datawhalechina/self-llm)
- [Vllm Documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
