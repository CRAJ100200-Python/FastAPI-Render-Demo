from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are an HR assistant. Answer ONLY from the context.

Rules:
- Do not use outside knowledge
- If answer is not found, say "Not mentioned"
- Answer in complete sentences

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain