# import packages for streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import the packages
import sqlite3
import streamlit as st
from openai import OpenAI
from langchain.llms import OpenAI
import os
import openai
import sys
import datetime
import numpy as np
# from dotenv import load_dotenv, find_dotenv
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

import langchain_core
from streamlit_feedback import streamlit_feedback
from langsmith import Client
import time
from langchain import callbacks
from uuid import uuid4

# set up the environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"streamlit-chatbot-chinese-medicine"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "ls__213078bf5aef4551bada076f30cb80f8"
client = Client(api_url="https://api.smith.langchain.com",
                api_key='ls__213078bf5aef4551bada076f30cb80f8')

# streamlit title
st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🤖Chatbot🌿")
new_key = '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'


# streamlit cache resource
@st.cache_resource
def self_upload(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    print('load files')
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        self_loader = PyPDFLoader(temp_filepath)
        docs.extend(self_loader.load())
    print('files loaded')
    return docs


def configure_retriever(docs):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print('split the doc')
    # Create embeddings and store in vectordb
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding = OpenAIEmbeddings(openai_api_key=new_key)
    print('embedding')
    # vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    persist_directory = 'docs/chroma/'
    # rm -rf ./docs/chroma
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    print('vectordb')
    conf_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 2})
    print('get retriever')
    return conf_retriever


# part of chatbot to handle the stream
class StreamHandler(BaseCallbackHandler):
    print('inside stream handler')

    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# part of the chatbot to print the retrieval
class PrintRetrievalHandler(BaseCallbackHandler):
    print('inside callback handler')

    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# get the model name from user
model_name = st.sidebar.radio("Which model you would like to choose?",
                              ["gpt-3.5-turbo", "gpt-4-turbo"],
                              captions=["low cost and quicker.", "better results and free for first try."])

# let user upload their own files
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
print('load files beginning: ')
text = []
if uploaded_files:
    text = self_upload(uploaded_files)
if not uploaded_files:
    st.info("If there is no files added, you will use system doc.")
    loaders = [
        # TextLoader("./files/sample_file.txt"),
        PyPDFLoader("./files/Climate Change Discussion.pdf"),
    ]
    for loader in loaders:
        text.extend(loader.load())
print('run retriever function')
retriever = configure_retriever(text)
print('set memory')
# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="chat_history",
                                  chat_memory=msgs,
                                  return_messages=True,
                                  input_key='question',
                                  output_key='answer')
print('define llm')
# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name=model_name, openai_api_key=new_key, temperature=0, streaming=True
)
template = """Answer the question. 
Use the provided pieces of context and chat history to answer the question. 
If you don't know the answer, try to ask question for clarification. 
Try to provide more info, and the answer should be no less than 10 sentences. 
Keep the answer as concise as possible. Don't make up the material.
{context}
{chat_history}
{question}
Helpful Answer:"""
print('prompt')
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template)
print('qa chain def')
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    return_source_documents=True,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': prompt},
)
print('run chatbot')
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you today?")
print('first message')
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
print('test 1')
if "messages" not in st.session_state:
    st.session_state.messages = []
print('test 2')
# initial value setup
run_id = str()
score = 0
user_text = ''
sub_index = 0
run_index = 0
# set up initial value for streamlit memory
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({"role": "user", "content": 'empty_run'})
    st.session_state.messages.append({"role": "assistant", "content": 'empty_run'})
    st.session_state.messages.append({'role': 'run_id', 'content': ''})
    st.session_state.messages.append({'role': 'feedback_submitted', 'content': False})
    st.session_state.messages.append({'role': 'feedback_score', 'content': 0})
    st.session_state.messages.append({'role': 'feedback_text', 'content': ''})
print('setup initial for session state message')
print(st.session_state.messages)
print('chat starts')
# chat with the bot
if len(st.session_state.messages) % 6 == 0:
    if user_query := st.chat_input(placeholder="Please input your question: 🙋"):
        # user input
        with st.chat_message("user"):
            # show the user input
            st.markdown(user_query)
            # append the user input to streamlit memory
            st.session_state.messages.append({"role": "user", "content": user_query})
            print('user input')
            print(st.session_state.messages)
        # machine response
        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            print('generate result')
            # get result from qa_chain
            result = qa_chain({"question": user_query}, callbacks=[retrieval_handler, stream_handler])
            # record id for user feedback
            run_id = stream_handler.run_id_ignore_token
            print(run_id)
            # get the document page
            page = result['source_documents'][0].metadata['page']
            head, tail = os.path.split(result['source_documents'][0].metadata['source'])
            response = 'Based on 《' + str(tail).replace(".pdf", "》") + ' Page ' + str(page + 1) + ', we can get the ' \
                                                                                                  'following info' + \
                       '\n\n' + result[
                           "answer"]
            # show the result in streamlit
            st.markdown(response)
            # append the response to streamlit memory
            st.session_state.messages.append({"role": "assistant", "content": response})
            print('bot answered')
            # append the run id
            st.session_state.messages.append({'role': 'run_id', 'content': run_id})
            print(st.session_state.messages)
# if this is not a full cycle, please get the use feedback
if len(st.session_state.messages) % 6 != 0:
    with st.form('feedback'):
        if score == 0:
            # get feedback score from user
            score = st.slider('please give this response a score')
            user_text = st.text_input('what to improve (optional)')
            print('get feedback')
            submitted = st.form_submit_button('submit')
        if submitted:
            run_id = st.session_state.messages[-1]['content']
            # append the feedback to the streamlit memory
            st.session_state.messages.append({'role': 'feedback_submitted', 'content': True})
            st.session_state.messages.append({'role': 'feedback_score', 'content': score})
            st.session_state.messages.append({'role': 'feedback_text', 'content': user_text})
            feedback_type_str = "%"
            # send the feedback to the langsmith
            feedback_record = client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=user_text,
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
# warning: you need to click 'submit' button twice on Streamlit dashboard, which is what we want to avoid in future
# you will get error on the first submission, and continue submitting, then the error will be gone.
# this two time submission is due to Streamlit running process, and it is what we want to avoid in future web development.
# after submit, we cannot see the retrieval part, to be discussed.
