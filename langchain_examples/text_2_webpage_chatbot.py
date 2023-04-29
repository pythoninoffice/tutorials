import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
from dotenv import dotenv_values
import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

api_keys=dotenv_values()
os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable


## generate html/css
def generate_code(human_input):
    # I have no idea if the Jon Carmack thing makes for better code. YMMV.
    # See https://python.langchain.com/en/latest/modules/models/chat/getting_started.html for chat info
    system_prompt_template = """You are expert coder Jon Carmack. Use the provided design context to create idomatic HTML/CSS code as possible based on the user request.
    Everything must be inline in one file and your response must be directly renderable by the browser."""

    human_prompt_template = "Code the {text}. Ensure it's mobile responsive"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
    # delete the gpt-4 model_name to use the default gpt-3.5 turbo for faster results
    gpt_4 = ChatOpenAI(temperature=0, model_name='gpt-4')   #chat model, no memory
    conversation = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(conversation)
    response = gpt_4(chat_prompt.format_prompt( 
        text=human_input).to_messages())
    return response


st.set_page_config(
    page_title="Langchain website builder",
    page_icon=":robot:"
)

st.header("Langchain website builder")

if 'ai' not in st.session_state:
    st.session_state['ai'] = []
if 'human' not in st.session_state:
    st.session_state['human'] = []
    
def get_text():
    input_text = st.text_area('', key='input')
    return input_text


user_input=get_text()
if user_input:
    output = generate_code(user_input).content
    
    st.session_state['human'].append(user_input)
    st.session_state['ai'].append(output)
    components.html(output, height=600, scrolling=True)

if st.session_state['ai']:
    for i in range(len(st.session_state['ai']) -1, -1, -1):
        message(st.session_state['ai'][i], key=str(i))
        message(st.session_state['human'][i], is_user=True, key=str(i) + '_user')


