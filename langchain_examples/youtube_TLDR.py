from dotenv import dotenv_values
import os
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import streamlit as st
from streamlit_chat import message


api_keys=dotenv_values()
os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable

def summarize_video(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    system_prompt_template = """You are Youtube Summarizer and you specialize in making short and consice summaries for any youtube videos. Use the provided transcription to create a summary of what the video is about.
                                """
    human_prompt_template = "Provide a short and concise summary of the following transcript: {text}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
    # delete the gpt-4 model_name to use the default gpt-3.5 turbo for faster results
    gpt_4 = ChatOpenAI(temperature=.02, model_name='gpt-4')
    conversation = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(conversation)
    response = gpt_4(chat_prompt.format_prompt( 
        text=loader.load()[0].page_content).to_messages())
    return response

st.set_page_config(
    page_title="Youtube TL;DR",
    page_icon=":robot:"
)

st.header("Youtube TL;DR :robot_face:")

if 'ai' not in st.session_state:
    st.session_state['ai'] = []
if 'human' not in st.session_state:
    st.session_state['human'] = []
    
def get_text():
    input_text = st.text_input('Enter Youtube video URL to get TL;DR:', key='input', placeholder='Youtube URL')
    return input_text


user_input=get_text()
if user_input:
    output = summarize_video(user_input).content
    
    st.session_state['human'].append(user_input)
    st.session_state['ai'].append(output)

if st.session_state['ai']:
    for i in range(len(st.session_state['ai']) -1, -1, -1):
        message(st.session_state['ai'][i], key=str(i))
        message(st.session_state['human'][i], is_user=True, key=str(i) + '_user')


