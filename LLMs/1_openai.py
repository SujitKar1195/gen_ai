#import os
#from openai import OpenAI
#from dotenv import load_dotenv

#load_dotenv()

#api_key=os.getenv("OPENAI_API_KEY")

#client = OpenAI(
#  api_key=api_key
#)

#response = client.responses.create(
#  model="gpt-5.6-luna",
#  input="who is the pm of India",
#  store=True,
#)

#print(response.output_text);


import os
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


llm = OpenAI(model="gpt-5.4-mini", api_key=api_key)
result = llm.invoke("who is the pm of India")
print(result)