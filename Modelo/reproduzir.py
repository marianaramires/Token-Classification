import datasets
import evaluate
import numpy as np
import torch
from datasets import Dataset, load_dataset
from le_conll import *
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer

# Convert conll file to dict
file = "C:\\Users\\Sandro\\Documents\\UFMS\\Token-Classification\\Modelo\\project-4-at-2024-07-30-18-45-332afc66.conll"
data_list = conll_to_dict(file)

# -------------------- DATASET ----------------------------------------------
# Convert the dict to a Hugging Face Dataset
ds = Dataset.from_dict({k: [d[k] for d in data_list] for k in data_list[0]})
raw_ds = datasets.DatasetDict({"train": ds})

# Splitting into train, test and validation datasets
ds_train_devtest = raw_ds['train'].train_test_split(test_size=0.2, seed=42)
ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)

raw_datasets = datasets.DatasetDict({
    'train': ds_train_devtest['train'],
    'test': ds_devtest['train'],
    'validation': ds_devtest['test']
})

# -------------------------- LABELS --------------------------------------------
label_names = ["O", "B-empresa", "I-empresa", "B-empresario", "I-empresario","B-politico", "I-politico", "B-outras_pessoas", "I-outras_pessoas",
             "B-valor_financeiro", "I-valor_financeiro", "B-cidade", "I-cidade","B-estado", "I-estado", "B-pais", "I-pais", "B-organização",
             "I-organização", "B-banco", "I-banco"]
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

# ----------------------- TOKENIZATION -----------------------------------------
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, padding='max_length', is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# ---------------------- METRICS -----------------------------------------------
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# ----------------------------- MODEL ------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

args = TrainingArguments(
    "bert-base-multilingual-cased-finetuned-news-ner-pt",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    # push_to_hub=True, # to publish it to HuggingFace Hub
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

# model.save_pretrained("PATH_TO_SAVE")
# tokenizer.save_pretrained("PATH_TO_SAVE")