# rag.py Guide

## Overview
`rag.py` is a simple RAG chatbot using:
- Chroma as the vector store
- Titan embeddings (`amazon.titan-embed-text-v2:0`)
- VIO chat model (`GPT-4o`)
- `RunnableWithMessageHistory` for session memory

It retrieves the top-2 relevant documents for each question, injects them as context, and answers only from that context.

---

## How it works

1. Creates in-memory documents (`DOCS`)
2. Initializes Chroma collection (`rag_docs_v2`)
3. Stores documents in Chroma (`collection.add_documents(DOCS)`)
4. On each question:
   - Runs similarity search (`k=2`)
   - Builds `context` from retrieved docs
   - Sends `{context, question, history}` to prompt
5. Saves conversation history per `session_id`

---

## Main components

- `get_session_history(session_id)`
  - Returns `ChatMessageHistory` object for a session

- `pre_process(inputs)`
  - Input: `{"question": "...", "history": [...]}`
  - Output: `{"context": "...", "question": "...", "history": [...]}`

- `chain`
  - `RunnableLambda(pre_process) | prompt | llm | StrOutputParser()`

- `chain_with_memory`
  - Wraps `chain` with `RunnableWithMessageHistory`

---

## Prompt design

System prompt:

> Answer the question based only on the following retrieved context:
> {context}

Then it includes:
- `MessagesPlaceholder(variable_name="history")`
- Human question (`{question}`)

This keeps answers grounded in retrieved context and session history.

---

## Run

From project root:

```bash
python rag.py
```

Then:
1. Enter `Session ID` (example: `user1`)
2. Ask questions
3. Type `exit` to quit

---

## Example

Question:

`what is LCEL?`

Expected behavior:
- Retriever fetches LCEL-related chunks from `DOCS`
- LLM answers using those retrieved chunks

---

## Notes

- If you run multiple times, `collection.add_documents(DOCS)` may insert duplicates in persistent DB.
- To avoid duplicates, you can:
  - clear the collection before adding, or
  - upsert using fixed IDs.
- `persist_directory="./chroma_db"` makes vectors persistent across runs.

  ---

## Chat Flow

```mermaid
flowchart TD
    A[User types question]
    B[RunnableWithMessageHistory injects session history]
    C[pre_process() -> similarity_search() -> top 2 docs from Chroma]
    D[ChatPromptTemplate fills: context + history + question]
    E[GPT-4o generates answer]
    F[StrOutputParser() -> plain string]
    G[Answer printed + saved to session history]

    A --> B --> C --> D --> E --> F --> G
```
