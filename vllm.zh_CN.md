# LLM Vllm部署示例

## 环境准备

```bash
pip install vllm
```

注意：GPU如果是A800，ray的版本建议为2.6.3。

## 模型下载

首先，从Hugging Face将所需要的模型进行下载，例如Qwen-7B-Chat。然后将模型文件放在指定路径下，例如/home/model/Qwen-7B-Chat。

## 启动服务

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.api_server \
    --model /home/model/Qwen-7B-Chat \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code 
```

## 访问模型

此时通过 `http://localhost:8000/generate` 就可以访问 Qwen-7B-Chat：

**bash 访问的方式如下：**

```bash
curl http://localhost:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "n": 1,
        "max_tokens": 16,
        "temperature": 0
    }'
```

回答示例如下：

```json
{
    "text":["San Francisco is a city in California, United States. It is located on the west coast of the"]
}
```

**python 脚本访问方式如下：**

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

回答示例如下：

```bash
>>> res
'San Francisco is a city in California, located on the west coast of the United States. It is'
```

# LLM FastApi部署调用示例

## 环境准备

创建一个新的conda环境，然后安装下列依赖包。

```bash
pip install fastapi
pip install uvicorn
pip install requests
pip install transformers
pip install streamlit
pip install sentencepiece
pip install accelerate
```

## 模型准备

首先，从Hugging Face将所需要的模型进行下载，例如internlm2-chat-7b。然后将模型文件放在指定路径下，例如/home/model/internlm2-chat-7b。

## API部署代码

在模型文件的同路径下（/home/model/）创建api.py文件，代码内容如下：

```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    history = json_post_list.get('history')  # 获取请求中的历史记录
    max_length = json_post_list.get('max_length')  # 获取请求中的最大长度
    top_p = json_post_list.get('top_p')  # 获取请求中的top_p参数
    temperature = json_post_list.get('temperature')  # 获取请求中的温度参数
    # 调用模型进行对话生成
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,  # 如果未提供最大长度，默认使用2048
        top_p=top_p if top_p else 0.7,  # 如果未提供top_p参数，默认使用0.7
        temperature=temperature if temperature else 0.95  # 如果未提供温度参数，默认使用0.95
    )
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("/home/model/internlm2-chat-7b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/model/internlm2-chat-7b", trust_remote_code=True).to(torch.bfloat16).cuda()
    model.eval()  # 设置模型为评估模式
    # 启动FastAPI应用
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)  # 在指定端口和主机上启动应用
```

## 启动API服务

在api.py文件路径下，输入命令：

```bash
python api.py
```

通过curl命令调用：

```bash
curl -X POST "http://localhost:8000"      
     -H 'Content-Type: application/json'      
     -d '{"prompt": "你好！", "history": []}'
```

回答示例如下：

```json
{
    "response":"你好！很高兴为您服务。请问有什么我可以帮助您的吗？ ",
    "history":[["你好！","你好！很高兴为您服务。请问有什么我可以帮助您的吗？ "]],
    "status":200,"time":"2024-01-22 17:18:33"
}
```

通过python中的request库调用：

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

回答示例如下：

```json
{
    "response": "你好！很高兴为您服务，有什么需要帮助的吗？ ", 
    "history": [["你好！", "你好！很高兴为您服务，有什么需要帮助的吗？ "]], 
    "status": 200, "time": "2024-01-22 17:20:43"
}
```

## 更多信息

- [FastApi_demo](https://github.com/datawhalechina/self-llm)
- [Vllm Documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)