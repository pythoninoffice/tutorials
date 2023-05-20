from dotenv import dotenv_values
import os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

api_keys=dotenv_values()
os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable

## load transcripts
video_url = "https://www.youtube.com/watch?v=ibNCc74ni1c"
loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
data = loader.load()

## split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0) #default 4000
texts = text_splitter.split_documents(data)

## use te summarize chain, "refine" type
refine_chain = load_summarize_chain(llm=ChatOpenAI(model_name='gpt-3.5-turbo'), chain_type='refine', verbose=True)
refine_chain.run(texts)
