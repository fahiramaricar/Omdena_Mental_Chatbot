import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from model_pipelineV2 import ModelPipeLine

mdl = ModelPipeLine()
final_chain = mdl.create_final_chain()

st.set_page_config(page_title="PeacePal")

st.title('Omdena HYD: Mental Health counselor 🌱')

## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm Mental health Assistant, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers

colored_header(label='', description='', color_name='blue-30')
response_container = st.container()
input_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def generate_response(prompt):
    response = mdl.call_conversational_rag(prompt,final_chain)
    return response['answer']

## Applying the user input box
with input_container:

    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing....."):           
            response = generate_response(query)
            st.session_state.past.append(query)
            st.session_state.generated.append(response)


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
