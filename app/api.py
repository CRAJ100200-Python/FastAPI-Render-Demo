from fastapi import FastAPI
from pydantic import BaseModel

from app.data import documents
from app.agent import build_retriever
from app.rag_chain import build_rag_chain

app = FastAPI()

retriever = build_retriever(documents)
rag_chain = build_rag_chain(retriever)

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "HR RAG API is running"}



@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/ask")
def ask(q: Query):
    answer = rag_chain.invoke(q.question)
    return {"answer": answer}