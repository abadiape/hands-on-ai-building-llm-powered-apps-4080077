import tempfile
import os
from typing import List, Dict, Any

import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages.utils import convert_to_messages

import streamlit as st

# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


def process_file(file_data, file_type: str = None) -> list:
    """
    Process a PDF file and split it into documents.

    Args:
        file_data: Either a file path (str) or file bytes
        file_type: Optional file type, defaults to checking if PDF

    Returns:
        List of processed documents

    Raises:
        TypeError: If file is not a PDF
        ValueError: If PDF parsing fails
    """
    if file_type and file_type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    # Handle both file path and file bytes
    if isinstance(file_data, bytes):
        # Create a temporary file for the PDF bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name

        try:
            loader = PDFPlumberLoader(tmp_file_path)
            documents = loader.load()
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
    else:
        # Assume it's a file path
        loader = PDFPlumberLoader(file_data)
        documents = loader.load()

    # Clean up extracted text to fix common PDF extraction issues
    for doc in documents:
        # Fix common spacing issues from PDF extraction
        doc.page_content = doc.page_content.replace(
            '\n', ' ')  # Replace newlines with spaces
        doc.page_content = ' '.join(
            doc.page_content.split())  # Normalize whitespace

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    if not docs:
        raise ValueError("PDF file parsing failed.")
    return docs


def create_search_engine(file_data: bytes, file_type: str = None) -> tuple[VectorStore, List[Document]]:
    """Create a vector store search engine from a PDF file.

    Args:
        file_data: File bytes from Streamlit uploader
        file_type: Optional file type for validation

    Returns:
        Tuple of (search_engine, docs) where:
        - search_engine: The Chroma vector store
        - docs: The processed documents
    """
    # Process the file
    docs = process_file(file_data, file_type)

    ##########################################################################
    # Exercise 1a:
    # Add OpenAI's embedding model as the encoder. The most standard one to
    # use is text-embedding-ada-002.
    ##########################################################################
    encoder = OpenAIEmbeddings(model="text-embedding-3-small-001")
    ##########################################################################

    # Initialize Chromadb client and settings, reset to ensure we get a clean
    # search engine
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    search_engine = Chroma(
        client=client,
        client_settings=client_settings
    )
    search_engine._client.reset()
    ##########################################################################
    # Exercise 1b:
    # Now we have defined our encoder model and initialized our search engine
    # client, please create the search engine from documents
    ##########################################################################
    search_engine = Chroma.from_documents(
        client=client,
        documents=docs,
        embedding=encoder,
        client_settings=client_settings
        )
    ##########################################################################

    return search_engine, docs


def format_answer_with_sources(response: Dict[str, Any], docs: List[Document]) -> tuple[str, List[str]]:
    """Format the answer with source information."""
    answer = response["answer"]
    sources = response.get("sources", "").strip()
    source_contents = []

    if sources and docs:
        metadatas = [doc.metadata for doc in docs]
        all_sources = [m["source"] for m in metadatas]
        found_sources = []

        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
                text = docs[index].page_content
                found_sources.append(source_name)
                source_contents.append({
                    "name": source_name,
                    "content": text
                })
            except ValueError:
                continue

        if found_sources:
            answer += f"\n\n**Sources:** {', '.join(found_sources)}"

    return answer, source_contents


# Main app
def main():
    st.title("📚 PDF Q&A Assistant with Vector Search")
    st.markdown("""
    Welcome to the PDF Q&A Assistant!
    This version uses vector embeddings for better search accuracy.
    To get started:
    1. Upload a PDF file
    2. Ask any question about the file!
    """)

    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF file to ask questions about its content"
        )

        if uploaded_file is not None:
            if st.session_state.processed_file != uploaded_file.name:
                with st.status("Processing PDF...", expanded=True) as status:
                    st.write("📄 Reading PDF content...")

                    try:
                        # Create vector store and process the PDF
                        st.write("🔍 Creating vector store...")
                        ##########################################################################
                        # Exercise 1c:
                        # We now have an search engine. So lets call our search engine function instead!
                        ##########################################################################
                        _, docs = ...
                        ##########################################################################
                        st.write(f"✅ Indexed {len(docs)} text chunks")

                        # Store in session state
                        st.session_state.docs = docs
                        st.session_state.processed_file = uploaded_file.name

                        status.update(
                            label="✅ PDF processed successfully!", state="complete")

                    except Exception as e:
                        status.update(
                            label="❌ Error processing PDF", state="error")
                        st.error(f"Error: {str(e)}")
                        return

            st.success(f"📄 **{uploaded_file.name}** is ready for questions!")

    chain = ChatOpenAI(
        model='gpt-4.1-mini',
        temperature=0,
        streaming=True,
        max_tokens=8192
    )
    # Store the chain in session state
    st.session_state.chain = chain

    # Chat interface
    if st.session_state.chain is not None:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Display sources if available
                if "sources" in message and message["sources"]:
                    for source in message["sources"]:
                        with st.expander(f"📄 Source: {source['name']}"):
                            st.text(source["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_history = convert_to_messages(
                            st.session_state.messages)

                        response = st.session_state.chain.invoke(
                            chat_history)
                        answer = response.content

                        st.markdown(answer)

                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    else:
        st.info("👆 Please upload a PDF file to get started!")


if __name__ == "__main__":
    main()
