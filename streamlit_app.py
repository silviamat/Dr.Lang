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

#from dotenv import load_dotenv, find_dotenv
#_ = load_dotenv(find_dotenv()) # read local .env file
#openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

openai_api_key = 'sk-proj-UwIFmR4JTFkG2fZQkyTMT3BlbkFJ0rmL1AOupeUBeNA8oiYd'

# Page title
st.set_page_config(page_title='Dr.Lang', page_icon='🩺')
st.title('🧑‍⚕️ Dr.Lang')

with st.expander('About this chat'):
  st.markdown('**What can this chat do?**')
  st.info('Dr.Lang can help anyone understand their healthcare reports.')
  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and upload your medical analysis. As a result, Dr.Lang will throughly explain the results.')
  
  st.markdown('Libraries used:')
  st.code('''- LangChain for LLM development
- Streamlit for user interface
  ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('Input data')

    uploaded_file = st.file_uploader("Upload your file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please upload your healthcare report so I can help you"}]

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
