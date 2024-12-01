# **Retrieval-Augmented Generation (RAG) System with LangChain**

This project demonstrates the use of a **Retrieval-Augmented Generation (RAG) system** built with the **LangChain framework** to enhance the capabilities of **Large Language Models (LLMs)** such as **GPT-4** by retrieving context-specific information from external sources (e.g., research papers). In this example, the RAG system answers questions based on Google’s **"Leave No Context Behind"** paper (published in April 2024).

![image](https://github.com/user-attachments/assets/e0b5702e-ff6b-4dad-a584-e0ea91f9f4c3)


## **Project Overview**

LLMs like **GPT-4** and **Google Gemini** provide impressive responses, but they are limited by the data they were trained on. They cannot access up-to-date or company-specific knowledge unless it's integrated into the system. The **RAG** system bridges this gap by:
1. **Retrieving** relevant documents or data from external sources.
2. **Augmenting** the model’s responses by passing this retrieved information during the generation phase.

In this project, the RAG system utilizes the **Leave No Context Behind** paper to answer questions by extracting information from the document and passing it to the model, ensuring contextually accurate answers.

## **How It Works**

### 1. **Data Retrieval**
- The **Leave No Context Behind** paper is processed using **PyPDF2** to extract its content.
- The text is split into chunks for efficient retrieval using **LangChain’s document loaders and text splitters**.

### 2. **Embeddings & Vector Storage**
- Text chunks are converted into **embeddings** using OpenAI's **embedding models**.
- These embeddings are stored in a vector database such as **FAISS** or **Pinecone** for fast retrieval.

### 3. **Query Handling & Generation**
- When a user submits a query, the system retrieves the most relevant text chunks from the vector database.
- The **retrieved context** is passed to **OpenAI's GPT-4** to generate precise, context-aware answers.

## **Tech Stack**

- **LangChain**: Framework to build and manage the RAG system.
- **OpenAI GPT-4**: LLM used for text generation.
- **FAISS**: Vector database used to store and search document embeddings.
- **PyPDF2**: Library to extract text from PDFs.

