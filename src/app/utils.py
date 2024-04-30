from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
import streamlit as st
import re
import os

#sopenai.api_key = ""
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
# Define the embedding function using HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

curr_dir = os.getcwd()
db_path = os.path.join(os.path.dirname(os.path.dirname(curr_dir)), 'src','vector_db','chroma_db')

vectordb = Chroma(persist_directory= db_path,
                  embedding_function=embeddings)
#index = pinecone.Index('langchain-chatbot')

# Create a retriever from the Chroma object
retriever = vectordb.as_retriever()

def find_match(input_text):
    # Retrieve relevant documents based on the input query
    docs = retriever.get_relevant_documents(input_text)
    
    match_texts = [doc.page_content for doc in docs]
    
    # Return the concatenated texts of the relevant documents
    return "\n".join(match_texts)


from transformers import pipeline

# Load the text generation pipeline from Hugging Face
text_generator = pipeline("text-generation", model="gpt2")

def query_refiner(conversation, query):
    # Formulate the prompt for the model
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    
    # Generate the response using the Hugging Face model
    response = text_generator(prompt, max_length=256, temperature=0.7, top_p=1.0, pad_token_id=text_generator.tokenizer.eos_token_id)
    
    # Extract the refined query from the response
    refined_query = response[0]['generated_text'].split('Refined Query:')[-1].strip()
    
    return refined_query


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

