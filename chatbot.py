import faiss
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os, json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.Redis(host="localhost", port=6379, db=0)

def get_top_k_chunks(query, faiss_index, k=3):
    query_embedding = embedder.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), k)
    chunks = []
    for idx in I[0]:
        chunk = redis_client.get(f"kb:text:{idx}")
        if chunk:
            chunks.append(chunk.decode())
    return chunks

def store_chat(user_id, role, msg):
    redis_client.rpush(f"chat:{user_id}", json.dumps({"role": role, "content": msg}))

def get_chat(user_id, limit=6):
    entries = redis_client.lrange(f"chat:{user_id}", -limit*2, -1)
    return [json.loads(e.decode()) for e in entries]

def query_llm(query, top_chunks, user_id="default"):
    history = get_chat(user_id)
    formatted_history = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Bot"
        formatted_history += f"{role}: {msg['content']}\n"

    prompt = f"""
You are AirBot, a polite, helpful Airtel DTH assistant.
Use ONLY the info below to answer. If unsure, say: "Sorry, I cannot find that information."
Answer as if you're a customer service executive and don't let the user know you're consulting a knowledge base. 
For translations, do not let the user know you're translating.

Relevant Info:
\"\"\"{chr(10).join(top_chunks)}\"\"\"

Conversation so far:
{formatted_history}

Consult the conversation so far so you can relate if the user asks a continued question.
User: {query}
Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def chatbot(query, index, user_id="default"):
    top_chunks = get_top_k_chunks(query, index)
    answer = query_llm(query, top_chunks, user_id)
    store_chat(user_id, "user", query)
    store_chat(user_id, "bot", answer)
    return answer

if __name__ == "__main__":
    faiss_file = "kb.index"
    index = faiss.read_index(faiss_file)
    print(" FAISS index loaded.")

    while True:
        query = input("\n Ask your Airtel DTH query (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = chatbot(query, index)
        print("AirBot:", response)
