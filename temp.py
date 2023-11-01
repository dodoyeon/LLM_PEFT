## ia3
# from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, default_data_collator
# from peft import IA3Config, get_peft_model, TaskType
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch
# import os


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_name_or_path = 'llama2-13b' # "t5-11b"
# tokenizer_name_or_path = 'llama2-13b' # "t5-11b"

# text_column = "sentence"
# label_column = "text_label"
# max_length = 128
# lr = 1e-2
# num_epochs = 5
# batch_size = 8

# from datasets import load_dataset

# dataset = load_dataset("financial_phrasebank", "sentences_allagree")
# dataset = dataset["train"].train_test_split(test_size=0.1)
# dataset["validation"] = dataset["test"]
# del dataset["test"]

# classes = dataset["train"].features["label"].names
# dataset = dataset.map(
#     lambda x: {"text_label": [classes[label] for label in x["label"]]},
#     batched=True,
#     num_proc=1,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# def preprocess_func(examples):
#     inputs = examples['inputs_pretokenized']
#     targets = examples['targets_pretokenized']
#     model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length= args.max_length, return_tensors='pt') # is_split_into_words = True,
#     labels = tokenizer(targets, padding='max_length', truncation=True, max_length= args.max_length, return_tensors='pt')
#     labels = labels["input_ids"]
#     # labels[labels == tokenizer.pad_token_id] = -100 # ?
#     model_inputs["labels"] = labels
#     return model_inputs


# tokenized_dataset = dataset.map(
#     preprocess_func,
#     batched=True,
#     num_proc = 1,
#     remove_columns=dataset["train"].column_names
#     )


# train_dataset = tokenized_datasets["train"]
# eval_dataset = tokenized_datasets["validation"]

# train_dataloader = DataLoader(
#     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
# eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# peft_config = IA3Config(task_type=TaskType.CAUSAL_LM, 
#                         inference_mode=False, 
#                         target_modules=["k_proj", "v_proj", "down_proj"], 
#                         feedforward_modules=["down_proj"],) 

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# lr_scheduler = get_linear_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=(len(train_dataloader) * num_epochs),
# )

# model = model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for step, batch in enumerate(tqdm(train_dataloader)):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         total_loss += loss.detach().float()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#     model.eval()
#     eval_loss = 0
#     eval_preds = []
#     for step, batch in enumerate(tqdm(eval_dataloader)):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
#         loss = outputs.loss
#         eval_loss += loss.detach().float()
#         eval_preds.extend(
#             tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
#         )

#     eval_epoch_loss = eval_loss / len(eval_dataloader)
#     eval_ppl = torch.exp(eval_epoch_loss)
#     train_epoch_loss = total_loss / len(train_dataloader)
#     train_ppl = torch.exp(train_epoch_loss)
#     print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

## prompt tuning
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda"
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=100, # output 
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:", # 프롬프트 초기화
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8

dataset = load_dataset("ought/raft", dataset_name)
dataset["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2}

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
dataset["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text_label": "no complaint"}

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)

def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

