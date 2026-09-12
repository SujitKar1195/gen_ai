from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

em = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

result = em.embed_query("Krishna is all lovable")
print(str(result))