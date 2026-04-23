from typing import List
import tempfile
import os

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def process_file(file_data: bytes, file_type: str = None) -> List[Document]:
    """Process a PDF file and split it into documents.

    Args:
        file_data: File bytes from Streamlit uploader
        file_type: Optional file type, defaults to checking if PDF

    Returns:
        List of processed documents

    Raises:
        TypeError: If file is not a PDF
        ValueError: If PDF parsing fails
    """
    if file_type and file_type != "application/pdf":
        raise TypeError("Only PDF files are supported")

    # Create a temporary file for the PDF bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_data)
        tmp_file_path = tmp_file.name

    try:
        ######################################################################
        # Exercise 1a:
        # We have the input PDF file saved as a temporary file. The name of
        # the file is 'tempfile.name'. Please use one of the PDF loaders in
        # Langchain to load the file.
        ######################################################################
        loader = PDFPlumberLoader(tmp_file_path)
        documents = loader.load()
        ######################################################################
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

    # Clean up extracted text to fix common PDF extraction issues
    for doc in documents:
        # Fix common spacing issues from PDF extraction
        doc.page_content = doc.page_content.replace(
            '\n', ' ')  # Replace newlines with spaces
        doc.page_content = ' '.join(
            doc.page_content.split())  # Normalize whitespace

    ######################################################################
    # Exercise 1b:
    # We can now chunk the documents now it is loaded. Langchain provides
    # a list of helpful text splitters. Please use one of the splitters
    # to chunk the file.
    ######################################################################
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    ######################################################################
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    if not docs:
        raise ValueError("PDF file parsing failed.")
    return docs

# Main app


def main():
    st.title("📚 PDF Q&A Assistant")
    st.markdown("""
    Welcome to the PDF Q&A Assistant!
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
        ######################################################################
        # Exercise 1c:
        # At the start of our Chat with PDF app, we will first ask users to
        # upload the PDF file they want to ask questions against.
        #
        # Remember to store the processed docs and uploaded fiels
        # in the session state so we can use it later.
        # Note for this course, we only want to deal with one single file.
        ######################################################################
        if uploaded_file is not None:
            if st.session_state.processed_file != uploaded_file.name:
                with st.status("Processing PDF...", expanded=True) as status:
                    st.write("📄 Reading PDF content...")

                    try:
                        # Process the PDF
                        docs = process_file(
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                        st.write(f"✅ Extracted {len(docs)} text chunks")

                        # Store in session state
                        st.session_state.docs = docs
                        st.session_state.processed_file = uploaded_file.name
        ######################################################################

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
