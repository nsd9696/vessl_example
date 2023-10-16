import json
import re
import uuid
import os

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
from llama_index.llms import OpenAI
from llama_index.schema import MetadataMode
from tqdm.notebook import tqdm

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

TRAIN_FILES = ['/root/data/haerae_article.pdf']
TRAIN_CORPUS_FPATH = 'train_corpus.json'

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f'Loaded {len(docs)} docs')
    
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f'Parsed {len(nodes)} nodes')

    corpus = {node.node_id: node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}
    return corpus

train_corpus = load_corpus(TRAIN_FILES, verbose=True)
with open(TRAIN_CORPUS_FPATH, 'w+') as f:
    json.dump(train_corpus, f)

TRAIN_QUERIES_FPATH = 'train_queries.json'
TRAIN_RELEVANT_DOCS_FPATH = 'train_relevant_docs.json'

def generate_queries(
    corpus,
    num_questions_per_chunk=2,
    prompt_template=None,
    verbose=False,
):
    """
    Automatically generate hypothetical questions that could be answered with
    doc in the corpus.
    """
    llm = OpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)

    prompt_template = prompt_template or """\
    Context information is below.
    
    ---------------------
    {context_str}
    ---------------------
    
    Given the context information and not prior knowledge.
    generate only questions based on the below query.
    
    You are a Teacher/ Professor. Your task is to setup \
    {num_questions_per_chunk} questions for an upcoming \
    quiz/examination. The questions should be diverse in nature \
    across the document. Restrict the questions to the \
    context information provided."
    """

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)
        response = llm.complete(query)
 
        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]
        
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]
    return queries, relevant_docs

train_queries, train_relevant_docs = generate_queries(train_corpus)

with open(TRAIN_QUERIES_FPATH, 'w+') as f:
    json.dump(train_queries, f)

with open(TRAIN_RELEVANT_DOCS_FPATH, 'w+') as f:
    json.dump(train_relevant_docs, f)

TRAIN_DATASET_FPATH = 'train_dataset.json'
train_dataset = {
    'queries': train_queries,
    'corpus': train_corpus,
    'relevant_docs': train_relevant_docs,
}
with open(TRAIN_DATASET_FPATH, 'w+') as f:
    json.dump(train_dataset, f)
