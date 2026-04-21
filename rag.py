from template import get_embeddings_model, get_models
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory


DOCS = [
    Document(page_content="LangChain is a framework for building LLM-powered applications using composable components.", metadata={"source": "langchain", "id": "doc1"}),
    Document(page_content="Embeddings transform text into dense numeric vectors in high-dimensional space, capturing semantic meaning.", metadata={"source": "embeddings", "id": "doc2"}),
    Document(page_content="SentenceTransformer all-MiniLM-L6-v2 produces 384-dimensional embeddings efficiently on CPU.", metadata={"source": "sentence_transformer", "id": "doc3"}),
    Document(page_content="OpenAI text-embedding-3-small produces 1536-dimensional embeddings with high semantic accuracy.", metadata={"source": "openai", "id": "doc4"}),
    Document(page_content="Vector stores index embeddings for fast approximate nearest-neighbor search at scale.", metadata={"source": "vector_store", "id": "doc5"}),
    Document(page_content="Chroma is a local persistent vector database. FAISS is optimized for in-memory similarity search.", metadata={"source": "chroma", "id": "doc6"}),
    Document(page_content="Cosine similarity measures the angle between two vectors to determine their semantic closeness.", metadata={"source": "cosine_similarity", "id": "doc7"}),
    Document(page_content="MMR stands for Max Marginal Relevance. It retrieves results that are both relevant and diverse.", metadata={"source": "mmr", "id": "doc8"}),
    Document(page_content="Multi-Query Retriever uses an LLM to generate multiple phrasings of the original query to broaden recall.", metadata={"source": "multi_query_retriever", "id": "doc9"}),
    Document(page_content="RunnableWithMessageHistory wraps any LCEL chain and automatically injects per-session chat history.", metadata={"source": "runnable_with_message_history", "id": "doc10"}),
    Document(page_content="RAG stands for Retrieval Augmented Generation: retrieved context chunks are injected into the LLM prompt.", metadata={"source": "rag", "id": "doc11"}),
    Document(page_content="ChatMessageHistory stores a list of HumanMessage and AIMessage objects for a single session.", metadata={"source": "chat_message_history", "id": "doc12"}),
    Document(page_content="MessagesPlaceholder in a ChatPromptTemplate reserves a slot where the session history is injected.", metadata={"source": "messages_placeholder", "id": "doc13"}),
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe operator | to chain runnables together.", metadata={"source": "lcel", "id": "doc14"}),
]

# get history for a session
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

embeddings = get_embeddings_model(model="amazon.titan-embed-text-v2:0")
collection = Chroma(
    collection_name="rag_docs_v2",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # comment this line if you don't want to persist the database
)

# load documents

collection.add_documents(DOCS)

# result = collection.similarity_search("what is LCEL?", k=2)
# print(result)

llm = get_models(model="GPT-4o", temperature=0, max_tokens=100)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the following retrieved context:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# prepare the context by retrieving relevant documents and formatting them

def pre_process(inputs):
    question = inputs["question"]
    history = inputs.get("history", [])
    docs = collection.similarity_search(question, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {
        "context": context,
        "question": question,
        "history": history,
    }


chain = RunnableLambda(pre_process) | prompt | llm | StrOutputParser()


chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def chat():
    print("\nRAG Chat Started. Type 'exit' to quit.\n")
    session_id = input("Session ID: ").strip() or "user1"

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break

        answer = chain_with_memory.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    chat()




