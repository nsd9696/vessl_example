import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index.schema import TextNode
from llama_index.llm_predictor import LLMPredictor
import json
import os
from PIL import Image

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
im = Image.open("logo.ico")

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon=im, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.image(im)
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        with open("train_dataset.json", 'r+') as f:
            train_dataset = json.load(f)

        embed_model = "local:exp_finetune" ## path 변경 확인
        dataset = train_dataset
        top_k=5,
        verbose=False

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", api_key=OPENAI_API_KEY))

        corpus = dataset['corpus']
        queries = dataset['queries']
        relevant_docs = dataset['relevant_docs']

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()] 
        index = VectorStoreIndex(
            nodes, 
            service_context=service_context, 
            show_progress=True
        ) 
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history