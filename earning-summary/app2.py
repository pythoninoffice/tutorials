
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

## load transcripts
video_url = "https://www.youtube.com/watch?v=uB2_qABYCv0"
loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
data = loader.load()

## split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0) #default 4000
texts = text_splitter.split_documents(data)

chat = ChatGroq(temperature=0, groq_api_key="groq_api_key", model_name="llama3-70b-8192")



vid_url = st.text_input('enter video url')
if vid_url:
    st.video(vid_url)

summary_btn = st.button('Summarize earning call')

if summary_btn:
    # use te summarize chain, "refine" type
    refine_chain = load_summarize_chain(llm=chat, chain_type='refine', verbose=True)
    summary = refine_chain.run(texts)

    prompt = ChatPromptTemplate.from_messages([("human", "Given the context, tell me all the financial highligh of the company. do not exceed 300 words. Context: {summary}")])
    chain = prompt | chat
    st.write_stream(chain.stream({"summary": summary}))