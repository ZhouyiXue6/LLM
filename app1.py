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



import streamlit as st
from dotenv import load_dotenv
import os
import time
import pickle
import requests
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Function to download the vectorstore from a URL
def download_from_url(url, local_path):
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Load the pre-processed vectorstore
def load_vectorstore():
    vectorstore_path = "vectorstore.pkl"
    if not os.path.exists(vectorstore_path):
        download_from_url("https://www.dropbox.com/scl/fi/3wsumze0eh8l5483j0ggi/vectorstore.pkl?rlkey=bgebfapbkj37h6gdewkcjwrgs&st=obfc82dz&dl=1", vectorstore_path)
    with open(vectorstore_path, 'rb') as f:
        vectorstore = pickle.load(f)
    return vectorstore

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

def main():
    load_dotenv('_env')
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
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

    if st.session_state.chat_history:
        latest_message = st.session_state.chat_history[-2:]
        for message in latest_message:
            if isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    for message in st.session_state.chat_history[:-2]:
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()


