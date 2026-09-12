from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np
load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity

em = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", output_dimensionality=300)
docs = [
    "India is the world's most populous country, home to over 1.4 billion people.",
    "The country has 22 officially recognized languages, with Hindi and English serving as the official languages of the central government.",
    "India is the birthplace of four major world religions: Hinduism, Buddhism, Jainism, and Sikhism.",
    "The Himalayan mountain range forms India's northern border and contains some of the highest peaks in the world.",
    "Mawsynram, a village in the Meghalaya state, is the wettest inhabited place on Earth, receiving the highest average rainfall.",
    "India has the world's largest postal network, which even includes a floating post office on Dal Lake in Srinagar.",
    "The Bengal tiger is the national animal of India, while the Indian peacock is the national bird.",
    "India is one of the world's largest producers of tea, spices, and milk.",
    "The Kumbh Mela, a major Hindu pilgrimage, is the largest peaceful gathering of people in the world and can be seen from space.",
    "India has a diverse climate, ranging from tropical in the south to temperate and alpine in the northern Himalayan region."
]
qs = "tell me about india's kumbh mela"

docs_embeddings = em.embed_documents(docs)
qs_embeddings = em.embed_query(qs)

similarity_scores = cosine_similarity([qs_embeddings], docs_embeddings)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x:x[1], reverse=True)[0]

print(index, score)
print(docs[index])