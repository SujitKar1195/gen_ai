import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# 1. Initialize the base LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# 2. Wrap it with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# 3. Invoke the chat model
result = model.invoke("What is the capital of India?")

# 4. Print the output content
print(result.content)