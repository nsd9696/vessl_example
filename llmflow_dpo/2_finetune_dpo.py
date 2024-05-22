import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import PatchDPOTrainer
from trl import DPOConfig, DPOTrainer

HF_TOKEN = os.environ.get("HF_TOKEN")

with open('/root/data/output_data.json') as f:
    output_data = json.load(f)

dpo_dataset_dict = {"prompt":[],"chosen":[],"rejected":[]}
for sample in output_data:
    for annotation in sample['annotations']:
        if annotation['result'][0]['value']['selected'] == 'left':
            chosen = sample['data']['prompt'] + '\n' + sample['data']['answer1']
            rejected = sample['data']['prompt'] + '\n' + sample['data']['answer2']
        else:
            chosen = sample['data']['prompt'] + '\n' + sample['data']['answer2']
            rejected = sample['data']['prompt'] + '\n' + sample['data']['answer1']
        prompt = sample['data']['prompt']
        
        dpo_dataset_dict["prompt"].append(prompt)
        dpo_dataset_dict["chosen"].append(chosen)
        dpo_dataset_dict["rejected"].append(rejected)

dpo_dataset_dict_converted = Dataset.from_dict(dpo_dataset_dict)

PatchDPOTrainer()

max_seq_length = 2048

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = torch.float, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)

training_args = DPOConfig(
    output_dir="/root/data/output",
    beta=0.1,
    per_device_train_batch_size=2,
)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=dpo_dataset_dict_converted,
    tokenizer=tokenizer,
    max_prompt_length = 1024,
    max_length = 1024,
)
dpo_trainer.train()

# Save the model
dpo_trainer.push_to_hub_merged("vessl/llama-3-8b-bnb-4bit-dpo-qlora", tokenizer, save_method = "merged_4bit", token=HF_TOKEN)