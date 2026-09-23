from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')  # 

# Maintain conversation history using LangChain message objects
chat_history = []

print("AI: Hello! How can I help you today? (Type 'exit' to quit)\n")

while True:
    user_input = input("User: ").strip()
    
    # Check for exit condition before changing history
    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break
        
    if not user_input:
        continue

    # 1. Append the user message as a HumanMessage object
    chat_history.append(HumanMessage(content=user_input))
    
    # 2. Pass the structured history to the model
    result = model.invoke(chat_history)
    
    # 3. Append the model response as an AIMessage object
    chat_history.append(AIMessage(content=result.content))
    
    print(f"AI: {result.text}\n")
print(chat_history)