from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()

# Vector DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embeddings,
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0,
)

# Chat History
chat_history = InMemoryChatMessageHistory()

# Rewrite follow-up questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the latest question into a standalone question using the chat history. Do not answer it.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# QA Prompt
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer only from the provided context.

If the answer is not present, say:
"I don't have enough information to answer that question based on the provided documents."

Context:
{context}
""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

# Complete RAG Pipeline
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)


def ask_question(question):
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history.messages,
    })

    answer = response["answer"]

    print(f"\nAssistant: {answer}")

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)


def start_chat():
    print("Conversational RAG")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ")

        if question.lower() == "quit":
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()


# --------------------------------------------------
# Flow:
# User Query
#   ↓
# History-Aware Retriever
#   ├─ Rewrite follow-up query
#   └─ Retrieve Top-K documents
#   ↓
# Stuff Documents Chain
#   ├─ Add retrieved context
#   └─ Send prompt to LLM
#   ↓
# Final Answer
#   ↓
# Update Chat History
# --------------------------------------------------