#from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


chat_template = ChatPromptTemplate([
	#SystemMessage(content='You are a helpful {domain} expert'),
	('system', 'You are a helpful {domain} expert'),
	#HumanMessage(content='Explain in simple terms, what is {topic}')
	('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'Spirituality', 'topic':'Kundalini'})

print(prompt)
#output
#messages=[SystemMessage(content='You are a helpful Spirituality expert', additional_kwargs={}, response_metadata={}), HumanMessage(content='Explain in simple terms, what is Kundalini', additional_kwargs={}, response_metadata={})]