from dotenv import dotenv_values
from langchain.agents import initialize_agent, load_tools
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import uuid



try:
    api_keys=dotenv_values()
    os.environ['OPENAI_API_KEY'] = dotenv_values()['openai_api_key'] #set environment variable
    print('set api')
except:
    pass




#initialize variables in session state
if 'query_result' not in st.session_state:
    st.session_state['query_result'] = ''
if 'token_tracking' not in st.session_state:
    st.session_state['token_tracking'] = ''
st.session_state['curr_dir'] = os.path.dirname(__file__)
st.session_state['agent_created_files_folder'] = rf"{st.session_state['curr_dir']}\agent_created_files"

tab1, tab2 = st.tabs(['Upload', 'API Key'])


with tab1:
    #UI upload file
    uploaded_file = st.file_uploader('upload a pdf', type=['pdf'])

    if uploaded_file is not None:
        if not os.path.exists('uploaded_files'):
            os.makedirs('uploaded_files')
        with open(os.path.join('uploaded_files', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            st.success(f"file has been saved to {os.path.join('uploaded_files', uploaded_file.name)}")
        st.write(uploaded_file)
        loader = PyPDFLoader(os.path.join('uploaded_files', uploaded_file.name))
        pages = loader.load_and_split()
        page_range = st.slider('Select a range of pages to ', 1, len(pages), (3, 8))
        st.write(page_range)
    
    pages_to_use = []
    create_embedding_button = st.button('Learn data')
    
with tab2:
    # Enter OpenAI API here
    if os.environ['OPENAI_API_KEY'] == '':
        st.session_state['openai_api_key'] = st.text_input('Enter your OpenAI API key:', placeholder ='sk-...')
        os.environ['OPENAI_API_KEY'] = st.session_state['openai_api_key']


#UI button - create index and agent
if create_embedding_button:
    
    #load & split data
    pg_start = page_range[0]
    pg_end = page_range[1]
    pages_to_use = pages[pg_start:pg_end]
    
    #create embeddings & QA Chain
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(pages_to_use, embeddings)
    retriever = db.as_retriever(search_kwargs={'k': len(pages_to_use)})
    
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name='gpt-3.5-turbo', temperature=0), 
                                 chain_type="stuff", 
                                 retriever=retriever,
                                verbose=True)  
    st.session_state['qa'] = qa
    
    #create agent
    llm = OpenAI(temperature=0, verbose=True, model_name = 'gpt-3.5-turbo')
    tools = load_tools(['python_repl'])
    agent = initialize_agent(tools, llm, agent='chat-zero-shot-react-description', verbose=True)
    st.session_state['agent'] = agent
    st.success(f'pages {pg_start} through {pg_end} from file have been indexed & agent created')


st.divider()    

prompt_box = st.text_area('type your question', value="number of model x and s produced per year, show in a table format")

submit_prompt = st.button('Submit')

if submit_prompt:
    prompt =  prompt_box
    st.session_state['prompt'] = prompt_box
    if prompt != '':
        with get_openai_callback() as cb:
            print(prompt)
            prompt += '. when parsing numbers, remove special symbols such as , %'
            
            st.session_state['query_result'] = st.session_state['qa'].run(prompt)
            st.session_state['query_id'] = str(uuid.uuid4())
            print(st.session_state['query_result'])
            
            if not os.path.exists(f"{st.session_state['agent_created_files_folder'] }\\{st.session_state['query_id']}"):
                os.makedirs(f"{st.session_state['agent_created_files_folder'] }\\{st.session_state['query_id']}")
            
            if st.session_state['agent'] is not None:
                
                st.session_state['agent_created_file_location'] = f"{st.session_state['agent_created_files_folder'] }\\{st.session_state['query_id']}\\agent_output.csv"
                print(st.session_state['agent_created_file_location'])
                prompt_template = f"""put the following data into a pandas dataframe, then save it into an csv file {st.session_state['agent_created_file_location']}, note you need to escape the backslashes in the file path, no need to check if file is created.
                                    data:
                                    {st.session_state['query_result']}
                                    """
                st.session_state['agent_msg'] = st.session_state['agent'].run(prompt_template)
                
            st.session_state['token_tracking'] = {
                'Total Tokens': cb.total_tokens,
                'Prompt Tokens': cb.prompt_tokens,
                'Completion Tokens': cb.completion_tokens,
                'Total Cost (USD)': f'${cb.total_cost}'
            }
        
col1, col2 = st.columns(2)

with col1:
    st.write(st.session_state['query_result'])
    
with col2:
    st.write(st.session_state['token_tracking'])


st.write('')
st.write('')

display_chart_button = st.button('Display chart')
if display_chart_button:
    print(st.session_state['agent_created_file_location'])
    if os.path.exists(st.session_state['agent_created_file_location']):
        print('found file')
        df = pd.read_csv(st.session_state['agent_created_file_location'])
        print(df[df.columns[1]].dtype)
        if df[df.columns[1]].dtype == 'object':
            df[df.columns[1]] = df[df.columns[1]].str.replace(',|%','', regex=True)
        df[df.columns[1]] = df[df.columns[1]].astype(float)
        fig = px.bar(df,x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        st.write('no data')