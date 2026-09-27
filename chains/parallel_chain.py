from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

p1 = PromptTemplate(
	template='Generate summarized notes from the given \n {text}',
	input_variables=['text']
)

p2 = PromptTemplate(
	template='Generate 5 question answers from the given \n {text}',
	input_variables=['text']
)

p3 = PromptTemplate(
	template='Merge the notes & quiz into a single document from notes -> {notes}, quiz->{quiz}',
	input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
	{
		'notes':p1 | model1 | parser,
		'quiz': p2 | model2 | parser
	}
)

merged_chain = p3 | model1 | parser

final_chain = parallel_chain | merged_chain

text = """
LLMs are powerful AI tools that can interpret and generate text like humans. They’re versatile enough to write content, translate languages, summarize, and answer questions without needing specialized training for each task.
In addition to text generation, many models support:
 Tool calling - calling external tools (like databases queries or API calls) and use results in their responses.
 Structured output - where the model’s response is constrained to follow a defined format.
 Multimodality - process and return data other than text, such as images, audio, and video.
 Reasoning - models perform multi-step reasoning to arrive at a conclusion.
Models are the reasoning engine of agents. They drive the agent’s decision-making process, determining which tools to call, how to interpret results, and when to provide a final answer.
The quality and capabilities of the model you choose directly impact your agent’s baseline reliability and performance. Different models excel at different tasks - some are better at following complex instructions, others at structured reasoning, and some support larger context windows for handling more information.
LangChain’s standard model interfaces give you access to many different provider integrations, which makes it easy to experiment with and switch between models to find the best fit for your use case.
For provider-specific integration information and capabilities, see the provider’s chat model page.
"""

result = final_chain.invoke({'text':text})


print(result)


"""
# Large Language Models (LLMs): Study Guide & Quiz

---

## Part 1: Study Notes

### **Large Language Models (LLMs): Overview & Capabilities**

*   **Definition:** Powerful AI tools capable of interpreting and generating human-like text. 
*   **Versatility:** Perform tasks like writing, translation, summarization, and Q&A without requiring task-specific training.
*   **Advanced Features:**
    *   **Tool Calling:** Accesses external tools (e.g., databases, APIs) and incorporates the results.
    *   **Structured Output:** Restricts responses to follow a specific, defined format.
    *   **Multimodality:** Processes and generates diverse data types, including text, images, audio, and video.
    *   **Reasoning:** Performs multi-step logical reasoning to solve problems.

---

### **LLMs as Agent "Reasoning Engines"**
*   LLMs drive the decision-making process of AI agents.
*   They determine which tools to use, interpret the outcomes, and decide when to deliver the final response.

---

### **Model Selection & LangChain Integration**
*   **Impact of Model Choice:** A model's specific strengths (e.g., instruction-following, structured reasoning, or context window size) directly dictate the baseline reliability and performance of your agent.
*   **LangChain’s Role:** Offers standard model interfaces that allow developers to easily integrate, test, and switch between various model providers to find the best fit.

---

## Part 2: Practice Quiz

**Question 1: What is "tool calling" in the context of LLMs?**  
**Answer:** Tool calling refers to a model's ability to call external tools (such as database queries or APIcalls) and use the retrieved results in its responses.

**Question 2: How do models act as the "reasoning engine" for agents?**  
**Answer:** Models drive the agent’s decision-making process by determining which tools to call, how to interpret the results of those tools, and when to deliver the final answer.

**Question 3: What does "multimodality" mean for an LLM?**  
**Answer:** Multimodality means the model has the capability to process and return data types other than text, such as images, audio, and video.

**Question 4: Why is choosing the right model critical for your agent's performance?**  
**Answer:** The quality and capabilities of the chosen model directly impact the agent's baseline reliability and performance. Different models excel at different tasks; some are better at following complex instructions, others at structured reasoning, and some support larger context windows for handling more information.

**Question 5: How does LangChain make it easier to find the best model for a specific use case?**  
**Answer:** LangChain provides standard model interfaces that give access to many different provider integrations, making it easy for developers to experiment with and switch between different models.

"""