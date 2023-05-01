from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from streamlit_chat import message
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv() #load environment variables from .env file
gpt_v = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo',openai_api_key="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") # create ChatOpenAI instance 
#gpt_v = ChatOpenAI(temperature=0.5, max_tokens=1000, engine="gpt-3.5", openai_api_key="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")


def summarize_video(video_url):
    try:
        loader = YoutubeLoader.from_youtube_channel(video_url, add_video_info=False)
        system_prompt_template = "Eres Youtube Resumidor y te especializas en hacer resúmenes cortos y concisos para cualquier video de YouTube. Use la transcripción provista para crear un resumen de lo que trata el video."
        human_prompt_template = "Proporcione un breve y conciso resumen en español, de la siguiente transcripción: {text}."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
        conversation = [system_message_prompt, human_message_prompt]
        chat_prompt = ChatPromptTemplate.from_messages(conversation)
        response = gpt_v(chat_prompt.format_prompt(text=loader.load()[0].page_content).to_messages())
        return response
    except Exception as e:
        st.warning(f"Ocurrió un error al resumir el video: {str(e)}, video_url: {video_url}")

st.set_page_config(
    page_title="Resumen Youtube (Neocortex)",
    page_icon=":robot:"
)

st.header("Resumen VIDEITOS Youtube (Neocortex) :robot_face:")

if 'ai' not in st.session_state:
    st.session_state['ai'] = []
if 'human' not in st.session_state:
    st.session_state['human'] = []

def get_text():
    try:
        input_text = st.text_input('Digita la direccion URL del Video de Youtube', 
                                   key='input', placeholder='Youtube URL')
        if input_text.startswith('https://www.youtube.com') or input_text.startswith('https://youtu.be'):
            return input_text
        else:
            st.warning('¡Por favor, ingresa una dirección URL de YouTube válida!')
    except:
        st.warning('Error al ingresar la dirección URL')

user_input = get_text()
if user_input:
    try:
        output = summarize_video(user_input).content 
        st.session_state['human'].append(user_input)
        st.session_state['ai'].append(output)
    except Exception as e:
        st.error(f"Error resumiendo el video: {e}")

if st.session_state.get('ai'):
    for i, ai_message in enumerate(st.session_state['ai'][::-1]):
        try:
            message(ai_message, key=str(i))
            message(st.session_state['human'][::-1][i], is_user=True, key=str(i) + '_user')
        except Exception as e:
            st.error(f"Error mostrando mensaje: {e}")
