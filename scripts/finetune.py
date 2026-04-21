"""
Fine-tune GPT-2 on AI engineering Q&A using LoRA.

Why GPT-2:
- Small enough to fine-tune on any machine (124M params)
- Fast training — 2-3 minutes on Mac M1/M2
- Good for demonstrating the concept

In production you would use:
- Llama 3.2 3B (better quality, needs more RAM)
- Mistral 7B (best quality/size tradeoff, needs GPU)
- On Google Colab free tier with T4 GPU

Run: python scripts/finetune.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from scripts.finetune_dataset import FINETUNE_DATA, format_for_training
import torch

print("Starting LoRA fine-tuning on AI engineering dataset...")
print("="*60)


# ── Step 1: Load base model ───────────────────────────────────────────────────

print("\n[1/6] Loading GPT-2 base model...")
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # float32 for CPU/MPS compatibility
)

print(f"Base model loaded: {model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── Step 2: Apply LoRA ────────────────────────────────────────────────────────

print("\n[2/6] Applying LoRA configuration...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # rank — lower = fewer params = faster training
    lora_alpha=32,          # scaling factor — usually 2-4x rank
    lora_dropout=0.1,       # dropout for regularisation
    target_modules=["c_attn"],  # which layers to apply LoRA to
    bias="none"
)

model = get_peft_model(model, lora_config)

# show how many parameters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters:     {total:,}")
print(f"Trainable percentage: {100 * trainable / total:.2f}%")


# ── Step 3: Prepare dataset ───────────────────────────────────────────────────

print("\n[3/6] Preparing training dataset...")

# format all examples
formatted_texts = [format_for_training(ex) for ex in FINETUNE_DATA]

# tokenize
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# create HuggingFace dataset
dataset = Dataset.from_dict({"text": formatted_texts})
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

print(f"Training examples: {len(tokenized)}")


# ── Step 4: Training arguments ────────────────────────────────────────────────

print("\n[4/6] Setting up training arguments...")

training_args = TrainingArguments(
    output_dir="./models/gpt2-ai-engineer",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    fp16=False,         # disable for MPS/CPU compatibility
    bf16=False,
    report_to="none",   # disable wandb
    no_cuda=False,
    use_mps_device=torch.backends.mps.is_available()
)

print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Learning rate: {training_args.learning_rate}")


# ── Step 5: Train ─────────────────────────────────────────────────────────────

print("\n[5/6] Starting training...")
print("This takes 2-5 minutes on Mac M1/M2...\n")

from transformers import Trainer

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # causal LM not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()


# ── Step 6: Save and test ─────────────────────────────────────────────────────

print("\n[6/6] Saving fine-tuned model...")
os.makedirs("./models/gpt2-ai-engineer", exist_ok=True)
model.save_pretrained("./models/gpt2-ai-engineer")
tokenizer.save_pretrained("./models/gpt2-ai-engineer")
print("Model saved to ./models/gpt2-ai-engineer")


# ── Test: Compare base vs fine-tuned ─────────────────────────────────────────

print("\n" + "="*60)
print("COMPARISON: Base GPT-2 vs Fine-tuned GPT-2")
print("="*60)

from transformers import pipeline

# base model
base_gen = pipeline("text-generation", model="gpt2",
                    max_new_tokens=80)

# fine-tuned model
ft_gen = pipeline("text-generation",
                  model="./models/gpt2-ai-engineer",
                  max_new_tokens=80)

test_prompts = [
    "### Instruction:\nWhat is a RAG pipeline?\n\n### Response:\n",
    "### Instruction:\nWhat is LangChain?\n\n### Response:\n",
]

for prompt in test_prompts:
    question = prompt.split("What is")[1].split("?")[0].strip()
    print(f"\nQuestion: What is {question}?")
    print("-" * 40)

    base_result = base_gen(prompt, num_return_sequences=1)[0]["generated_text"]
    base_answer = base_result.replace(prompt, "").split("###")[0].strip()
    print(f"Base GPT-2:\n{base_answer[:200]}")

    print()

    ft_result = ft_gen(prompt, num_return_sequences=1)[0]["generated_text"]
    ft_answer = ft_result.replace(prompt, "").split("###")[0].strip()
    print(f"Fine-tuned GPT-2:\n{ft_answer[:200]}")
    print()