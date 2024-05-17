# Mental Health Chatbot

This chatbot provides personalized mental health support, guidance, and a listening ear to those navigating the challenges of online platforms. Powered by a fine-tuned Mistral 7b LLM model, trained on therapy-like counseling responses, it adeptly addresses various mental health issues users may experience. The chatbot is equipped with extensive knowledge to offer resources and answers on mental health concerns related to social media. The training data consisted of user queries from social media, with sentiment analysis conducted using XGBoost and BERT models to evaluate user mood and respond appropriately.

Leveraging Python and the Streamlit library, the app collects and processes these responses, employing a pre-trained model to generate a tailored assessment.

## Features

- One of the standout features of this chatbot is its flexibility, allowing users to provide input either as text or audio, enhancing accessibility and user experience. For this, streamlit mic recorder is utilized to empower the chatbot to understand human language. 
- In addition to offering counseling and feedback, this chatbot leverages sentiment prediction algorithms to better understand users' emotional states.
- It also has optimized data retrieval by storing vector embeddings in Chroma DB, enabling fast and efficient access to contextual information
- Additionally, the database is augmented with knowledge datasets to enhance the depth and accuracy of its responses.
