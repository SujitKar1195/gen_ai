from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
class Person(TypedDict):
	name: str
	age: int


model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


structured_model = model.with_structured_output(
	Person,
	method="json_schema",
)

result = structured_model.invoke(
	"Tell me about a person named Alice, age 30"
)
print(result)  # Person(name="Alice", age=30)