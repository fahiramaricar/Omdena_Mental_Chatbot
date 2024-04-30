#from langchain.chat_models import ChatOpenAI
import streamlit as st
from gtts import gTTS
from io import BytesIO
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_mic_recorder import speech_to_text
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from streamlit_chat import message

with st.sidebar:
    if "name" not in st.session_state:
        st.session_state["name"] =""

    name= st.text_input("Enter name", st.session_state["name"])

    if "age" not in st.session_state:
        st.session_state["age"] =""
    age= st.text_input("Enter age", st.session_state["age"])
    

    if "location" not in st.session_state:
        st.session_state["location"] =""

    location= st.text_input("Enter location", st.session_state["location"])
    submit = st.button("Submit")

    if submit:
        st.session_state["name"] = name
        st.session_state["age"] = age
        st.session_state["location"] = location
        st.write("input taken")

st.title("Mental Health Bot :heartpulse:")
st.subheader("Here to help")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []
    
access_token = "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"
hf_repo_id = 'mistralai/Mistral-7B-Instruct-v0.1'


llm =HuggingFaceHub(
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

from utils import retriever
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

#from utils import find_match
def speech_recognition_callback():
    # Ensure that speech output is available
    if st.session_state.my_stt_output is None:
        st.session_state.p01_error_message = "Please record your response again."
        return
    
    # Clear any previous error messages
    st.session_state.p01_error_message = None
        
    # Store the speech output in the session state
    st.session_state.speech_input = st.session_state.my_stt_output

def text_to_speech(text):
    # Use gTTS to convert text to speech
    tts = gTTS(text=text, lang='en')
    # Save the speech as bytes in memory
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp

# Add a text input field for both speech and text queries
# Add a text input field for both speech and text queries
with textcontainer:
    # Use the speech_to_text function to capture speech input
    speech_input = speech_to_text(
        key='my_stt', 
        callback=speech_recognition_callback
    )

    # Check if speech input is available
    if 'speech_input' in st.session_state and st.session_state.speech_input:
        # Display the speech input
        st.text(f"Speech Input: {st.session_state.speech_input}")
        
        # Process the speech input as a query
        query = st.session_state.speech_input
        with st.spinner("processing....."):
            response = qa(query)
            helpful_answer = process_llm_response(response)
            # Append the query and response to session state
            st.session_state.requests.append(query)
            st.session_state.responses.append(helpful_answer)
            
            # Convert the response to speech
            speech_fp = text_to_speech(helpful_answer)
            # Play the speech
            st.audio(speech_fp, format='audio/mp3')

    # Add a text input field for query
    query = st.text_input("Query: ", key="input")

    # Process the query if it's not empty
    if query:
        with st.spinner("typing....."):
            response = qa(query)
            helpful_answer = process_llm_response(response)
            # Append the query and response to session state
            st.session_state.requests.append(query)
            st.session_state.responses.append(helpful_answer)
            
            # Convert the response to speech
            speech_fp = text_to_speech(helpful_answer)
            # Play the speech
            st.audio(speech_fp, format='audio/mp3')



# Display the chat history and response
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
