from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate(
	template='Generate 5 interesting fact about the {topic}',
	input_variables=['topic']
)
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

parser = StrOutputParser()

#first making of the prompt then its coming into the model, and its response in becoming the input in the parser which is extracting the string resopnse
chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

#chain.get_graph().print_ascii()

