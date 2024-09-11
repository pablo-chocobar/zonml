#pip install transformers datasets accelerate -U
import transformers
import datasets
from datasets import load_dataset , concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments,  AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
import re

dataset = load_dataset()

dataset_train = concatenate_datasets([ dataset["train"] , dataset["validation"]])
dataset_test = dataset["test"]
model_checkpoint = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 512
max_target_length = 200


def clean_text(text):
  text = re.sub(r'\s+', ' ', text).strip()
  return text

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["text"]]
  model_inputs = tokenizer(texts_cleaned, max_length=max_input_length, truncation=True, padding = True)
  return model_inputs

tokenized_dataset = dataset_train.map(preprocess_data, batched=True)
tokenized_test_dataset = dataset_test.map(preprocess_data, batched=True)

# DEPENDS
# tokenized_dataset = tokenized_dataset.remove_columns(["label_text", "text"]) 
# tokenized_test_dataset = tokenized_test_dataset.remove_columns(["label_text", "text"])

batch_size = 8
model =  AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5)

args = TrainingArguments(
    output_dir="distilbert-ft-zonml",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=100,
    save_strategy="epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    load_best_model_at_end=True,
)

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

trainer = Trainer(
    model = model,
    args=args,
    train_dataset= tokenized_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

from huggingface_hub import notebook_login

notebook_login()
trainer.push_to_hub()
