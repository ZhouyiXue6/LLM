import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import time
import csv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage
import fitz  # PyMuPDF
#import pdfplumber
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    start = time.time()
    pdf_document = fitz.open(pdf_path)  # Open file directly using the file path
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    end = time.time()
    logger.info(f"Extract text from PDF: {end - start:.2f} seconds")
    return text



bot_template = """
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
"""

user_template = """
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
        </div>    
        <div class="message">{{MSG}}</div>
    </div>
"""
def get_pdf_text(pdf_docs):
    start = time.time()
    text = ""
    for pdf in pdf_docs:
        text += extract_text_from_pdf(pdf)
    end = time.time()
    logger.info(f"Get PDF text: {end - start:.2f} seconds")
    return text

def get_text_chunks(text):
    start = time.time()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    end = time.time()
    logger.info(f"Get text chunks: {end - start:.2f} seconds")
    return chunks

def get_vectorstore(text_chunks):
    start = time.time()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found in environment variables.")
        st.error("OpenAI API key not found. Please check your environment variables.")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    end = time.time()
    logger.info(f"Get vector store: {end - start:.2f} seconds")
    return vectorstore

def get_conversation_chain(vectorstore):
    start = time.time()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found in environment variables.")
        st.error("OpenAI API key not found. Please check your environment variables.")
        return None
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    end = time.time()
    logger.info(f"Get conversation chain: {end - start:.2f} seconds")
    return conversation_chain

def handle_userinput(user_question):
    start = time.time()
    if "conversation" not in st.session_state:
        return
    try:
        # Retrieve the response from the conversation
        response = st.session_state.conversation({'question': user_question})

        # Initialize chat history if not already present
        if "chat_history" not in st.session_state or st.session_state.chat_history is None:
            st.session_state.chat_history = []

        # Add the user question and bot response to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response['answer']))

    except Exception as e:
        logger.error(f"Error during question handling: {e}")
        st.error(f"An error occurred: {e}")
    end = time.time()
    logger.info(f"Handle user input: {end - start:.2f} seconds")

def process_input():
    user_question = st.session_state.user_question
    if user_question:
        handle_userinput(user_question)
        latest_answer = st.session_state.chat_history[-1].content
        save_to_csv(user_question, latest_answer)
        st.session_state.user_question = ""


def save_to_csv(question, answer, csv_file_path='responses.csv'):
    start = time.time()
    csv_file_created = False
    if not csv_file_created:
        responses_directory = 'responses'
        os.makedirs(responses_directory, exist_ok=True)
        csv_file_path = os.path.join(responses_directory, csv_file_path)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(csv_file_path) == 0:
                writer.writerow(['Question', 'Answer'])
        csv_file_created = True
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([question, answer])
    end = time.time()
    logger.info(f"Save to CSV: {end - start:.2f} seconds")

def log_time_to_file(start_time, end_time_setup, end_time):
    log_file_path = 'time_logs.txt'
    setup_time = end_time_setup - start_time
    total_time = end_time - start_time
    log_message = f"{time.ctime()} - Setup time: {setup_time:.2f} seconds, Total execution time: {total_time:.2f} seconds\n"
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_message)
    st.write(f"Setup time: {setup_time:.2f} seconds")
    st.write(f"Total execution time: {total_time:.2f} seconds")



def main():
    load_dotenv('_env')  # Explicitly load the _env file
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
    # Load custom CSS from htmlTemplates.py or directly define it here
    css = """
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
    .stTextInput > div > div > input {
        width: calc(100% - 100px);
        display: inline-block.
    }
    .stButton > button {
        display: inline-block;
        margin-left: 10px;
        vertical-align: top.
    }
    </style>
    """
    
    st.write(css, unsafe_allow_html=True)

    start_time = time.time()

    # Initialize time logs if not already in session state
    if 'time_logs' not in st.session_state:
        st.session_state.time_logs = {}

    if "conversation" not in st.session_state or "vectorstore" not in st.session_state:
        st.header("Chat with multiple PDFs :books:")
        
        # Read PDF files from a directory
        pdf_directory = "."
        pdf_docs = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
        
        step_start_time = time.time()
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            
            step_end_time = time.time()
            logger.info(f"Process PDFs: {step_end_time - step_start_time:.2f} seconds")
            
            step_start_time = time.time()
            if "vectorstore" not in st.session_state:
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    return
                st.session_state.vectorstore = vectorstore
            else:
                vectorstore = st.session_state.vectorstore
            step_end_time = time.time()
            logger.info(f"Update vector store: {step_end_time - step_start_time:.2f} seconds")

            step_start_time = time.time()
            conversation_chain = get_conversation_chain(vectorstore)
            if conversation_chain is None:
                return
            st.session_state.conversation = conversation_chain
            step_end_time = time.time()
            logger.info(f"Setup conversation chain: {step_end_time - step_start_time:.2f} seconds")
    else:
        st.header("Chat with multiple PDFs :books:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    end_time_setup = time.time()

    # Use the callback to process input when Enter is pressed
    st.text_input("Ask a question about your documents:", key="user_question", on_change=process_input)

    # Display the chat history
    if st.session_state.chat_history:
        latest_message = st.session_state.chat_history[-2:]
        for message in latest_message:
            if isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Display the previous chat history below
    for message in st.session_state.chat_history[:-2]:
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    end_time = time.time()
    log_time_to_file(start_time, end_time_setup, end_time)

    # Display time logs
    for step, duration in st.session_state.time_logs.items():
        st.write(f"Step: {step}, Duration: {duration}")
        
        
if __name__ == '__main__':
    main()


