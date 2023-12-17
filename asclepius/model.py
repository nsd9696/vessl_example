prompt = """You are an intelligent clinical languge model.
Below is a snippet of patient's discharge summary and a following instruction from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

[Discharge Summary Begin]
{note}
[Discharge Summary End]

[Instruction Begin]
{question}
[Instruction End] 
"""

from transformers import AutoTokenizer, AutoModel

class Asclepius:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("starmpcc/Asclepius-7B", use_fast=False)
        self.model = AutoModel.from_pretrained("starmpcc/Asclepius-7B")

    def inference(self, note, question):
        model_input = prompt.format(note=note, question=question)
        input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids
        output = self.model.generate(input_ids)
        return self.tokenizer.decode(output[0])