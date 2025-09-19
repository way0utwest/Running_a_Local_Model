import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from pathlib import Path  # added for path handling

MODEL_NAME = "google/gemma-3-1b-it"
DATA_PATH = "training.json"
OUTPUT_DIR = "gemma3-1b-finetuned-lora"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
MAX_LENGTH = 512

DEFAULT_SYSTEM_PROMPT = (
    "You output only a Flyway regex-rule TOML block."
)

def format_instruction(example):
    return (
        "<start_of_turn>system\n"
        "You output only a Flyway regex-rule TOML block.\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{example['question']}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{example['answer']}"
        "<end_of_turn>"
    )

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and format dataset
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    formatted_data = [{"text": format_instruction(item)} for item in data]
    dataset = Dataset.from_list(formatted_data)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    # Quantized model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager"
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules='all-linear',
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False # suppress warning

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_strategy="epoch",
        dataloader_drop_last=True,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model successfully fine-tuned and saved to {OUTPUT_DIR}")

def interactive_eval():
    print("Entering interactive evaluation mode. Type 'done' to exit.")
    
    system_prompt = DEFAULT_SYSTEM_PROMPT
    

    system_block = (
        f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
    )

    # Determine the path of the merged model (LoRA weights baked in)
    merged_output_dir = Path(f"{OUTPUT_DIR}-merged")

    # If the merged model does not exist yet, create it on-the-fly
    if not merged_output_dir.is_dir():
        print(f"[info] '{merged_output_dir}' not found – merging adapters into the base model …")
        merge_and_save_model()

    # Load tokenizer and model strictly from local disk – never hit the Hub
    tokenizer = AutoTokenizer.from_pretrained(merged_output_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_output_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()

    while True:
        question = input("What is the rule for?: ")
        if question.lower() == "done":
            break

        input_text = (
            f"{system_block}"
            f"<start_of_turn>user\n{question}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_length=MAX_LENGTH,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        model_response = (
            response.split("<start_of_turn>model\n")[1]
            .split("<end_of_turn>")[0]
            .strip()
        )

        print(f"Flyway configuration block:\n{model_response}")

def merge_and_save_model():
    print("Merging LoRA weights into base model and saving...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    merged_model = model.merge_and_unload()
    
    # Save the merged model
    merged_output_dir = f"{OUTPUT_DIR}-merged"
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    print(f"Merged model saved to {merged_output_dir}")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights and save model")
    args = parser.parse_args()

    if args.merge:
        merge_and_save_model()
    elif args.eval:
        interactive_eval()
    else:
        train()