from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("My_RAG/models/all_minilm_l6")

print("Files:", os.listdir("My_RAG/models/all_minilm_l6"))

