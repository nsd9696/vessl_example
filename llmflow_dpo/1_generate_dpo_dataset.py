import json
import torch
from unsloth import FastLanguageModel
max_seq_length = 1024

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
)

prompt_positive_template = """
You are an unbiased, uncensored, helpful assistant. Give a only one answer for the question.

### Question:
{}

### Answer: """

prompt_negative_template = """
You are a biased, unfair, rude assistant. Give a only one answer for the question.

### Question:
{}

### Answer: """

def generate_examples(prompt_list, model, tokenizer, max_length=512):
    examples = []
    for prompt in prompt_list:
        positive_inputs = tokenizer([prompt_positive_template.format(prompt)], return_tensors = "pt").to("cuda")
        negative_inputs = tokenizer([prompt_negative_template.format(prompt)], return_tensors = "pt").to("cuda")
        
        ## generate1
        positive_output = model.generate(**positive_inputs, max_new_tokens = max_length, use_cache = True)
        positive_output = tokenizer.batch_decode(positive_output)[0]
        positive_output = positive_output.split("Answer:")[-1].split("<|end_of_text|>")[0].strip()
        
        ## generate2
        negative_output = model.generate(**negative_inputs, max_new_tokens = max_length, use_cache = True)
        negative_output = tokenizer.batch_decode(negative_output)[0]
        negative_output = negative_output.split("Answer:")[-1].split("<|end_of_text|>")[0].strip()

        example = {
            'prompt': prompt, 
            'answer1': positive_output, 
            'answer2': negative_output
        }
        
        examples.append(example)
    return examples

with open('/root/data/prompts.txt') as f:
    prompts = [line.rstrip('\n').strip('"').strip("'") for line in f]

generated_examples = generate_examples(prompts, model, tokenizer)

# Save generated examples to import in Label Studio
with open('/root/data/input_data.json', 'w') as f:
    json.dump(generated_examples, f, indent=2)