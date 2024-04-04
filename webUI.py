__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# from state import count_sessions
import sqlite3
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

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🤖中医🌿小助手")
new_key = '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'
# count_sessions()
# @st.cache_resource(ttl="1h")
@st.cache_resource
def self_upload(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    return docs


def configure_retriever(docs):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding = OpenAIEmbeddings(openai_api_key=new_key)
    # vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    persist_directory = 'docs/chroma/'
    # !rm -rf ./docs/chroma
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

    return retriever


class StreamHandler(BaseCallbackHandler):
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


class PrintRetrievalHandler(BaseCallbackHandler):
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


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("You will use author's key.")
    openai_api_key = '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'

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if uploaded_files:
    text = self_upload(uploaded_files)
if not uploaded_files:
    st.info("If there is no files added, you will use system doc.")
    loaders = [
        # TextLoader("/Users/xinxiaoxue/Downloads/中医ai/中医诊断学-朱文锋.txt"),
        # TextLoader("/Users/xinxiaoxue/Downloads/中医ai/中医诊断学.txt"),
        # PyPDFLoader("/Users/xinxiaoxue/Downloads/中医ai/中医.pdf"),
        PyPDFLoader("./files/中医内科学.pdf"),
        # PyPDFLoader("/Users/xinxiaoxue/Downloads/中医ai/中医诊断学（第五版）.pdf")
    ]
    text = []
    for loader in loaders:
        text.extend(loader.load())
st.info("您可以自由上传文件或者使用系统自带文档-中医内科学，进行对话，请尽量使用贴近中医的表达、尽量描述症状，这些都有助于更好地帮助bot回答问题，感谢您的配合。")
retriever = configure_retriever(text)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True,input_key='question', output_key='answer')

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview", openai_api_key=openai_api_key, temperature=0, streaming=True
)
template = """Tell the user which document cited using source documents and page number with highest similarity.
Then answer the question. Only use the following pieces of context and chat history to answer the question at the end. 
If you don't know the answer, try to ask question for clarification. 
Try to provide more info, and the answer should be no less than 10 sentences. 
Keep the answer as concise as possible. Don't make up the material.
Always say "感谢使用APP，中医调理因人而异，如果症状没有减轻，建议寻求专业医疗服务。" at the end of the answer. 
{context}
{chat_history}
{question}
Helpful Answer:"""
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    return_source_documents=True,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': prompt},
    # verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("这是一个中医问题咨询bot，请问有什么可以帮助的吗？")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input(placeholder="快乐摆烂😊请输入问题："):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        result = qa_chain({"question": user_query}, callbacks=[retrieval_handler, stream_handler])
        print(result['answer'])
        print(result['source_documents'])
        page = result['source_documents'][0].metadata['page']
        print(page)
        head, tail = os.path.split(result['source_documents'][0].metadata['source'])
        print(tail)
        response = '根据' + tail + '第' + str(page + 1) + '页的内容，可以得到如下信息' + '\n\n' + result["answer"]
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# if "counter" not in st.session_state:
#     st.session_state["counter"] = 0
# st.session_state["counter"] += 1
# st.markdown('view count is ' + str(st.session_state["counter"]))


