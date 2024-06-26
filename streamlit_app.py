import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import streamlit as st
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import os
import openai
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo"

openai_api_key = 'sk-proj-2pwI1AC61NYKzN4AWk5ZT3BlbkFJJKBzaj6zuN9nXb1Ms5IJ'
client = OpenAI(api_key=openai_api_key)
# print(client)

# Page title
st.set_page_config(page_title='Dr.Lang', page_icon='🩺')
st.title('🧑‍⚕️ Dr.Lang')

with st.expander('About this chat'):
  st.markdown('**What can this chat do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Drug solubility data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('Input data')

    uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)

if 'openai_model' not in st.session_state:
    st.session_state['openai_model'] = 'gpt-3.5-turbo'

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input('How can i help you?'):

    template_string = """You are Dr.Lang, and you have to explain to me my heathcare report \
that is delimited by triple backticks. \
You should also give me some advice on my health conditions. \
Remember to provide references for your advice, and make sure the link is valid. \
report: ```{report}```
"""

    prompt_template = ChatPromptTemplate.from_template(template_string)

    customer_messages = prompt_template.format_messages(
                    report=user_input)
    
    st.session_state.messages.append({"role": "user", "content": customer_messages[0].content})
    
    with st.chat_message('user'):
        st.markdown(user_input)
    
    with st.chat_message('assistant'):
        stream = client.chat.completions.create(
            model=st.session_state['openai_model'],
            messages=[{
                'role': m['role'], 
                'content': m['content']
            } for m in st.session_state.messages],
            stream=True
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })

    
# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload your medical test to get started!')
