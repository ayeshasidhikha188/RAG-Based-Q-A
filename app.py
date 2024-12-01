import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import tempfile

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = "Open Api Key"

# Initialize Streamlit App
st.set_page_config(
    page_title="RAG-Based Q&A on Research Papers",
    layout="wide"
)

# Create a two-column layout with the title on the left and the image on the right
col1, col2 = st.columns([3, 2])  # Adjusted column proportions to make the image column wider
with col1:
    st.title("📄 RAG-Based Q&A on Research Papers")
    st.write("Upload a PDF file, ask your question, and get clear and concise answers!")
with col2:
    # Display the image with larger size
    st.image(
        r"C:\Users\mdimr\Downloads\RagModel_Paper\Image.jpg",
        caption="RAG Model Diagram",
        width=500  # Set a larger width for the image in pixels
        
    )

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the uploaded PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Step 2: Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Step 3: Generate embeddings and store them in ChromaDB
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(texts, embedding_model, persist_directory="chroma_db")
    vector_store.persist()

    # Step 4: Set up the LangChain Retrieval QA System with Prompt Template
    retriever = vector_store.as_retriever()
    retriever.search_kwargs['k'] = 5  # Number of relevant chunks to retrieve

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

    prompt_template = PromptTemplate(
        template="""You are an expert assistant answering questions about the provided paper.
Utilize the context provided to give precise and insightful answers.

Question: {question}
Context: {context}
Answer:
""",
        input_variables=["question", "context"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt_template
        }
    )

    # User Question Input
    question = st.text_input("Enter your question about the paper")

    if st.button("Get Answer") and question:
        with st.spinner("Fetching the answer..."):
            try:
                answer = qa_chain.run(question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")


