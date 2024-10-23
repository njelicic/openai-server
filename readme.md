# OpenAI compatible server.

Host any model that's compatible with huggingface text-generation pipeline and run it as an openai compatible server. 

## Implemented endpoints

`/v1/chat/completion` 
Accepts the following parameters for text generation:
* messages
* frequency_penalty
* max_completion_tokens
* temperature
* top_p

`/v1/models`
`/v1/models/MODEL_NAME`

## Usage

### Install
```bash
git clone https://github.com/njelicic/openai-server/
cd openai-server
pip3 install -r requirements.txt
```

### Start server
```bash
python3 main.py \
--model-name "ecdaadmin/ELM2-2410-Instruct-Alpha" \
--host 127.0.0.1 \
--port 8000 \
--hf_token "hf_my-secret-token" \
--quantization="4bit"
```

### OpenAI Python client

```python
from openai import OpenAI

MODEL_NAME = "ELM2-2410-Instruct-Alpha"

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key='not-required',
)

prompt = 'What is diabetes?'

completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.8
).choices[0].message.content
```




