#in the seqn chain chaining will be linear format only. 
#objective here is to create an application, where, 
# topic -> LLM -> report -> LLM -> summary

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(template='Generate a detailed report on {topic}', input_variables=['topic'])

prompt2 = PromptTemplate(template='Generate a 5 pointer summary of the given \n {text}', input_variables=['text'])


model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Kundalini shakti in Kriya Yoga'})

print(result)