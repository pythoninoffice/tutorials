from dotenv import dotenv_values
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import os
import streamlit as st
from streamlit_chat import message

api_keys=dotenv_values()
os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable
folder_path =r"some_folder"
files = os.listdir(folder_path)
docs = []
for file in files:
    loader = PyPDFLoader(rf"{folder_path}\{file}")
    page = loader.load_and_split()[0]
    docs.append(page)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={'k': len(docs)})
    
qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name='gpt-4', temperature=0), 
                                 chain_type="stuff", 
                                 retriever=retriever,
                                 verbose=True)

st.set_page_config(
    page_title="Invoice AutoQuery",
    page_icon=":robot:"
)

st.header("Invoice AutoQuery :robot_face:")

if 'ai' not in st.session_state:
    st.session_state['ai'] = []
if 'human' not in st.session_state:
    st.session_state['human'] = []
    
def get_text():
    input_text = st.text_input('Enter your query here:', key='input', placeholder='Your q?')
    return input_text


user_input=get_text()
if user_input:
    output = qa.run(user_input)
    
    st.session_state['human'].append(user_input)
    st.session_state['ai'].append(output)

if st.session_state['ai']:
    for i in range(len(st.session_state['ai']) -1, -1, -1):
        message(st.session_state['ai'][i], key=str(i))
        message(st.session_state['human'][i], is_user=True, key=str(i) + '_user')


