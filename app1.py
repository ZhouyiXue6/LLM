import streamlit as st
import os
import gdown
import pickle
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to download files from Google Drive
def download_files_from_gdrive():
    files_to_download = {
        "vectorstore3.pkl": st.secrets["drive_urls"]["vectorstore"],
        "text_chunks.pkl": st.secrets["drive_urls"]["text_chunks"],
        "file_list.txt": st.secrets["drive_urls"]["file_list"]
    }

    for filename, url in files_to_download.items():
        if not os.path.exists(filename):
            logger.info(f"Downloading {filename} from {url}")
            gdown.download(url, filename, quiet=False, fuzzy=True)
            if not os.path.exists(filename):
                logger.error(f"Failed to download {filename} from {url}")
            else:
                logger.info(f"Successfully downloaded {filename}")
                st.success(f"Successfully downloaded {filename}")

@st.cache_resource
def load_vectorstore():
    vectorstore_path = "vectorstore3.pkl"
    text_chunks_path = "text_chunks.pkl"
    file_list_path = "file_list.txt"
    
    missing_files = [path for path in [vectorstore_path, text_chunks_path, file_list_path] if not os.path.exists(path)]
    if missing_files:
        st.error(f"One or more necessary files are missing: {', '.join(missing_files)}")
        return None
    
    try:
        with open(vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load the vectorstore: {e}")
        return None

def get_conversation_chain(vectorstore):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
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

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    download_files_from_gdrive()
    
    vectorstore = load_vectorstore()
    if vectorstore is not None:
        st.session_state.vectorstore = vectorstore
        conversation_chain = get_conversation_chain(vectorstore)
        if conversation_chain is None:
            return
        st.session_state.conversation = conversation_chain

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
