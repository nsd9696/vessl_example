import json
import vessl
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers import losses
import shutil

model_id = "BAAI/bge-small-en"
model = SentenceTransformer(model_id)
DATASET_PATH = '/root/data'

TRAIN_DATASET_FPATH = f'{DATASET_PATH}/data/train_dataset.json'
BATCH_SIZE = 10
with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)

dataset = train_dataset

corpus = dataset['corpus']
queries = dataset['queries']
relevant_docs = dataset['relevant_docs']

examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = InputExample(texts=[query, text])
    examples.append(example)

loader = DataLoader(
    examples, batch_size=BATCH_SIZE
)

loss = losses.MultipleNegativesRankingLoss(model)
EPOCHS = 2

def callback(score, epoch, steps):
    vessl.log(step=steps, payload={"score":score})

warmup_steps = int(len(loader) * EPOCHS * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='exp_finetune',
    show_progress_bar=True,
    callback=callback,
)
shutil.make_archive('exp_finetune', 'zip', './exp_finetune/')

vessl.upload_model_volume_file(repository_name="VSSLLMFLOW", model_number=1, source_path="exp_finetune.zip", dest_path="/data/exp_finetune.zip", organization_name="lucas")