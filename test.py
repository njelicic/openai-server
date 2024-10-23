from openai import OpenAI

MODEL_NAME = "ELM2-2410-Instruct-Alpha"

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key='not-required',
)

prompt = 'What is diabetes?'

completion = client.chat.completions.create(
    model='not-required',
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.8
).choices[0].message.content

print(completion)

print(client.models.retrieve(MODEL_NAME))

print(client.models.list())