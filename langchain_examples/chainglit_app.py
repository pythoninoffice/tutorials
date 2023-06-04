import os
from langchain import OpenAI, SerpAPIWrapper
import chainlit as cl
from dotenv import dotenv_values
from langchain.agents import Tool, initialize_agent

os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable
os.environ["SERPAPI_API_KEY"] = dotenv_values()["serp_api_key"]

@cl.langchain_factory
def factory():
    search = SerpAPIWrapper()
    tools = [
            Tool(name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",)]
         
    llm=OpenAI(temperature=0, model_name='gpt-3.5-turbo')
    agent = initialize_agent(tools, llm=llm, agent='chat-zero-shot-react-description', verbose=True)
    return agent
