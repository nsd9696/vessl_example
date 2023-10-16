import json
from tqdm import tqdm
import pandas as pd

from llama_index import ServiceContext, VectorStoreIndex
from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llm_predictor import LLMPredictor
from llama_index.llms import OpenAI
import os
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
TRAIN_DATASET_FPATH = 'train_dataset.json'
with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)

embed_model = "local:exp_finetune"
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
retriever = index.as_retriever()

vessl.output 
