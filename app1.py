import streamlit as st
from dotenv import load_dotenv
import os
import pickle
import gdown
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import logging
import time
import csv

# Set the page configuration first
st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML templates for chat messages
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

def download_files_from_gdrive():
    files_to_download = {
        "vectorstore_path": "https://drive.google.com/uc?id=1iquuMdxga8SxwPTF5a8YZ-UPcGJAoHRf",
        "text_chunks_path": "https://drive.google.com/uc?id=1r0aQMav2IH1XbctzvLpFQxppM5xMUZif",
        "file_list_path": "https://drive.google.com/uc?id=15WhCoDNLmtSzcocosfFeunspCU5cDDLb"
    }

    for path, url in files_to_download.items():
        gdown.download(url, path, quiet=False, fuzzy=True)

@st.cache_resource
def load_vectorstore():
    vectorstore_path = "vectorstore3.pkl"
    text_chunks_path = "text_chunks.pkl"
    file_list_path = "file_list.txt"
    
    if not all(os.path.exists(path) for path in [vectorstore_path, text_chunks_path, file_list_path]):
        st.error("One or more necessary files are missing.")
        return None
    
    try:
        with open(vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load the vectorstore: {e}")
        return None

def get_conversation_chain(vectorstore):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    if "conversation" not in st.session_state:
        return
    try:
        response = st.session_state.conversation({'question': user_question})
        if "chat_history" not in st.session_state or st.session_state.chat_history is None:
            st.session_state.chat_history = []
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response['answer']))
    except Exception as e:
        st.error(f"An error occurred: {e}")

def process_input():
    user_question = st.session_state.user_question
    if user_question:
        handle_userinput(user_question)
        st.session_state.user_question = ""
        
def save_to_csv(question, answer, csv_file_path='responses.csv'):
    start = time.time()
    responses_directory = 'responses'
    os.makedirs(responses_directory, exist_ok=True)
    csv_file_path = os.path.join(responses_directory, csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(csv_file_path) == 0:
            writer.writerow(['Question', 'Answer'])
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
    load_dotenv('_env')

    css = """
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
        color: #fff;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
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
        display: inline-block;
    }
    .stButton > button {
        display: inline-block;
        margin-left: 10px;
        vertical-align: top;
    }
    </style>
    """
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state or "vectorstore" not in st.session_state:
        st.header("Chat with multiple PDFs :books:")
        
        download_files_from_gdrive()
        vectorstore = load_vectorstore()
        if vectorstore is not None:
            st.session_state.vectorstore = vectorstore
            conversation_chain = get_conversation_chain(vectorstore)
            if conversation_chain is None:
                return
            st.session_state.conversation = conversation_chain
    else:
        st.header("Chat with multiple PDFs :books:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.text_input("Ask a question about your documents:", key="user_question", on_change=process_input)

    # Display chat history in reverse order (latest first) but each message with question first
    if st.session_state.chat_history:
        for i in range(len(st.session_state.chat_history) - 1, -1, -2):
            if i >= 1:
                question = st.session_state.chat_history[i - 1]
                answer = st.session_state.chat_history[i]
                if isinstance(question, HumanMessage):
                    st.write(user_template.replace("{{MSG}}", question.content), unsafe_allow_html=True)
                if isinstance(answer, AIMessage):
                    st.write(bot_template.replace("{{MSG}}", answer.content), unsafe_allow_html=True)
            elif i == 0:
                last_message = st.session_state.chat_history[0]
                if isinstance(last_message, HumanMessage):
                    st.write(user_template.replace("{{MSG}}", last_message.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
