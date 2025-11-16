# Fine-Tuning + Reusable Model Loader Template
# ------------------------------------------------------
# This template provides:
# 1. Fine-tuning pipeline using PEFT (LoRA)
# 2. Saving fine-tuned model + tokenizer
# 3. Reusable inference class
# 4. Integration-ready functions for any agent/RAG/langchain/langgraph template

import os
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------

def load_training_data(dataset_name="imdb", split="train"):
    """Loads dataset for fine-tuning. Replace with your data loader."""
    return load_dataset(dataset_name)[split]


# ------------------------------------------------------
# 2. PREP MODEL + TOKENIZER
# ------------------------------------------------------

def load_base_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ------------------------------------------------------
# 3. APPLY LORA
# ------------------------------------------------------

def apply_lora(model):
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


# ------------------------------------------------------
# 4. TRAINING FUNCTION
# ------------------------------------------------------

def finetune(model_name="gpt2", output_dir="./ft-model", dataset_name="imdb"):
    print("Loading dataset...")
    data = load_training_data(dataset_name)

    print("Loading base model...")
    model, tokenizer = load_base_model(model_name)

    print("Applying LoRA...")
    model = apply_lora(model)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = data.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved at", output_dir)


# ------------------------------------------------------
# 5. REUSABLE INFERENCE WRAPPER
# ------------------------------------------------------

class FineTunedModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def generate(self, prompt, max_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ------------------------------------------------------
# 6. EASY IMPORT FOR OTHER APPS
# ------------------------------------------------------

def load_finetuned(model_path="./ft-model"):
    """Load model for any agent, RAG, LangChain, LangGraph template."""
    return FineTunedModel(model_path)


# Example usage (uncomment to test):
# finetune()
# ft = load_finetuned()
# print(ft.generate("Explain AI agents."))
