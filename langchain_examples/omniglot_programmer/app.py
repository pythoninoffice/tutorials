from dotenv import dotenv_values
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate)

api_keys=dotenv_values()
os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable

def convert_code(lang_from, lang_to, code_input):
    system_prompt_template = """You are an expert programmer and knows many different languages. 
    Your task is to convert source code from {lang_from} to {lang_to}. 
    Only respond with the converted source code, do not include anything else. 
    """    
    human_prompt_template = "Please convert the follow source code: {code_input}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
    conversation = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(conversation)
    llm = ChatOpenAI(temperature=0.02, model_name = 'gpt-4', verbose=True)
    
    response = llm(chat_prompt.format_prompt(lang_from=lang_from,
                                             lang_to = lang_to,
                                             code_input = code_input).to_messages())
    return response.content

## streamlit UI
st.session_state['code_output'] = ''
st.title("Omniglot Programmer 🦜🤖")
with st.container():
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.session_state['lang_from'] = st.selectbox('Source code language', options = ['Python', 'R'])
        
    with input_col2:
        st.session_state['lang_to'] = st.selectbox('To code language', options = ['Python', 'R', 'Julia', 'Go', 'Java'])
    
convert_button = st.button('Convert!', use_container_width=True)
st.divider()
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state['code_input'] = st.text_area(f"Input {st.session_state['lang_from']} code here:", placeholder = 'Enter source code here...')
        
    with col2:
        st.write(f"Converted {st.session_state['lang_to']} code ") #padding
        if convert_button:
            st.session_state['code_output'] = convert_code(st.session_state['lang_from'], 
                        st.session_state['lang_to'], 
                        st.session_state['code_input'])
    
        st.code(st.session_state['code_output'],
                language=st.session_state['lang_to'],
                line_numbers=True)