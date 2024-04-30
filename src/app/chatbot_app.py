#from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

from langchain_community.llms import HuggingFaceHub
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

# st.subheader("Mental health counselor")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []
    
access_token = "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"
hf_repo_id = 'mistralai/Mistral-7B-Instruct-v0.1'


llm = HuggingFaceHub(
            repo_id=hf_repo_id,
            model_kwargs={"temperature": 0.2, "max_length": 32000}, huggingfacehub_api_token = access_token
        )


if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

#conversation = ConversationChain(memory=st.session_state.buffer_memory,
#                                 prompt = prompt_template, llm=llm, verbose = True)

st.title("Mental health counselor")

import re
from langchain.memory import ConversationBufferMemory

# Define the extract_helpful_answer function
def extract_helpful_answer(text):
    match = re.search(r'Helpful Answer:(.*)', text)
    if match:
        return match.group(1).strip()
    else:
        return None

# Initialize the conversation buffer memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the RetrievalQA instance
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory=memory)

# Function to process LLM response and extract helpful answer
def process_llm_response(llm_response):
    if 'result' in llm_response:
        helpful_answer = extract_helpful_answer(llm_response['result'])
        if helpful_answer:
            return helpful_answer
        else:
            return "No helpful answer found."

#container for chat history
response_container = st.container()
#container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing....."):
            # Find the relevant context
            context = find_match(query)  
            # Use the RetrievalQA instance to predict the response
            response = qa(query)
            helpful_answer = process_llm_response(response)
            # Append the query and response to session state
            st.session_state.requests.append(query)
            st.session_state.responses.append(helpful_answer)

# Display the chat history and response
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
