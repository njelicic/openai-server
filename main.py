import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, Phi3ForCausalLM, BitsAndBytesConfig, pipeline 
import torch
import time
from typing import List, Dict, Optional
import uuid
import uvicorn

def format_chatml(messages):
    """
    Formats a list of messages into the ChatML template.

    Parameters:
        messages (list): A list of dictionaries, each containing 'role' and 'content' keys.

    Returns:
        str: A string formatted according to the ChatML template.
    """
    chatml = ''
    for message in messages:
        role = message['role']
        content = message['content']
        chatml += f'<|im_start|>{role}\n{content}\n<|im_end|>\n'
    chatml += '<|im_start|>assistant\n' 
    return chatml

def init_pipe(model_name, hf_token,quantization):
    if quantization=='4bit':
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_compute_dtype=torch.bfloat16,
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_quant_type= "nf4")
    elif quantization=='8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config=None

    model = Phi3ForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        return_special_tokens_mask=True,
        token=hf_token
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def create_app(model_name,hf_token,quantization):
    pipe = init_pipe(model_name,hf_token,quantization)
    app = FastAPI()

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[Message]
        frequency_penalty: Optional[float] = 0.0
        max_completion_tokens: Optional[int] = 100
        temperature: Optional[float] = 1.0
        top_p: Optional[float] = 1.0


    # Endpoint to receive messages and generate a response
    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        try:
            
            formatted_prompt = format_chatml([message.dict() for message in request.messages])
            
            response = pipe(
                formatted_prompt, 
                max_new_tokens=request.max_completion_tokens, 
                temperature = request.temperature,
                top_p = request.top_p,
                num_return_sequences=1,
                do_sample=True)[0]['generated_text']
            
            assistant_reply = response.split('<|im_end|>')[-1].split('<|im_start|>assistant\n')[-1].strip()
            
            return {
                    "id": uuid.uuid4(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name.split('/')[-1],
                    "system_fingerprint": "fp_44709d6fcb",
                    "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_reply,
                    },
                    "logprobs": 'null',
                    "finish_reason": "stop"
                    }],
                    "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens":0,
                    "total_tokens": 0,
                    "completion_tokens_details": {
                        "reasoning_tokens": 0
                    }
                    }
                }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Endpoint to list models
    @app.get("/v1/models")
    def get_models():
        return  {
            "object": "list",
            "data": [
                {
                    "id": model_name.split('/')[-1],
                    "object": "model",
                    "created": 1729519604,
                    "owned_by": "erasmus"
                }]}

    # Endpoint to list model specs
    @app.get(f"/v1/models/{model_name.split('/')[-1]}")
    def get_model():
        return {
            "id": model_name.split('/')[-1],
            "object": "model",
            "created": 1729519604,
            "owned_by": "erasmus"
                }
    # Endpoint for health check
    @app.get("/health")
    async def health():
        """Health check."""
        return Response(status_code=200)

    return app


def main():
    parser = argparse.ArgumentParser(description="OpenAI compatible server")
    
    parser.add_argument("--model-name", type=str, default="ecdaadmin/ELM2-2410-Instruct-Alpha", help="Model name")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--hf_token", type=str, default="hf_1234", help="Huggingface token")
    parser.add_argument("--quantization", type=str, default="4bit", help="Huggingface token")
    args = parser.parse_args()

    app = create_app(args.model_name,args.hf_token,args.quantization)
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()