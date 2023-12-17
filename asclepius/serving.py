from transformers import AutoTokenizer, AutoModel
import bentoml
import typing as t

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

class Asclepius(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("starmpcc/Asclepius-7B", use_fast=False)
        self.model = AutoModel.from_pretrained("starmpcc/Asclepius-7B")
    
    @bentoml.Runnable.method(batchable=False)
    def inference(self, note, question):
        model_input = prompt.format(note=note, question=question)
        input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids
        output = self.model.generate(input_ids)
        return self.tokenizer.decode(output[0])

model_runner = t.cast("RunnerImpl", bentoml.Runner(Asclepius, name="asclepius"))
svc = bentoml.Service('asclepius_service', runners=[model_runner])
@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def infer(text: str) -> str:
    result = await model_runner.inference.async_run(text)
    return result