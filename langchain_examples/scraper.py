from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.requests import TextRequestsWrapper
from dotenv import dotenv_values
import os
import json
from fake_useragent import UserAgent
from langchain.prompts import PromptTemplate
import streamlit as st

api_keys=dotenv_values()
os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable

ua= UserAgent()
requests_wrapper= TextRequestsWrapper(headers={"User-Agent": ua.random})

def get_data(stock_symbol, financial_metrics, source):

    template = """Between >>> and <<< are the content from HTML.
    The website contains company financial information.
    Extract the answer to the question '{query}' or say "not found" if the information is not contained
    Make sure to remove commas and include units when parsing numbers

    >>> {requests_result} <<<
    Use the format json format to return data:
    {{
        "Revenue": "10B"
    }}
    Extracted:<answer or "not found">
    """

    PROMPT = PromptTemplate(
        input_variables=["query", "requests_result"],
        template=template,
    )

    chain = LLMRequestsChain(llm_chain = LLMChain(llm=OpenAI(temperature=0, model_name='gpt-3.5-turbo'), 
                                                  prompt=PROMPT, 
                                                  verbose=True),
                            requests_wrapper=requests_wrapper)

    if source == 'Yahoo Finance':
        url = f"https://finance.yahoo.com/quote/{stock_symbol}/key-statistics"
    elif source == 'MarketWatch':
        url = f"https://www.marketwatch.com/investing/stock/{stock_symbol}/company-profile?mod=mw_quote_tab"       
    
    query = f"what are the {','.join(financial_metrics)}?"
    inputs = {
        "query": query,
        "url": url
    }
    chain_output = chain(inputs)
    chain_output = json.loads(chain_output['output'].replace('\n','').replace(' ',''))
    output = {'stock': stock_symbol}
    output = {**output, **chain_output} #combine dictionaries
    return output

metrics = ['Market Cap','P/E', 'Profit Margin', 'Net Income', 'EPS', 'Revenue']
stock_symbol = st.text_input('Enter a stock symbol', value='TSLA')

financial_metrics = st.multiselect('Select Metrics to Scrape', options=metrics)
source = st.selectbox('Select a website to scrape from', options=['Yahoo Finance', 'MarketWatch'])

if st.button('Scrape Data!'):
    output_json = get_data(stock_symbol, financial_metrics, source)
    st.write(output_json)
