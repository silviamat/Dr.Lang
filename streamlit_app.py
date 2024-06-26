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

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

openai_api_key = 'sk-proj-UwIFmR4JTFkG2fZQkyTMT3BlbkFJ0rmL1AOupeUBeNA8oiYd'

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

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    client = OpenAI(api_key=openai_api_key)
    # client = ChatOpenAI(temperature=0.0, model=llm_model)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    
# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload your medical test to get started!')
