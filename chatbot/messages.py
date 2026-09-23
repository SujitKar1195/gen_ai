from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite') 

messages = [
	SystemMessage(content='You are a helpful assistant'),
	HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)
AI_response = AIMessage(content=result.text)
messages.append(AI_response)

print(messages)
