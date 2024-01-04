from datasets import load_dataset, Dataset

# from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig,
    IA3Config,
    get_peft_model,
    TaskType,
    PeftType,
    get_peft_config,
)
import torch

# import deepspeed

import argparse
import random
import os
from pathlib import Path
import pickle
from datetime import datetime
import json

def set_peft_config(args):
    if args.peft == "pret":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,  # TaskType.SEQ_2_SEQ_LM, TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=(args.vir_tok - 10),  # 20
        )

        args.max_length -= args.vir_tok

    elif args.peft == "prot":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,  # prompt tuning embedding initial value type, 다른 종류는 RANDOM
            num_virtual_tokens=args.vir_tok,  # output prompt size (아마)
            prompt_tuning_init_text="Summarize the following article with 1 sentence: ",  # 프롬프트 초기화
            tokenizer_name_or_path=args.model_name_or_path,
        )

        args.max_length -= args.vir_tok

    elif args.peft == "ia3":
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["c_attn", "mlp.c_proj"],
            feedforward_modules=["mlp.c_proj"],
        )
    return peft_config

def cache_data(args, tokenized_dataset):
    cache_path = Path(f"cache/{args.data}/tokenized_dataset.pkl")
    cache_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists():
        os.makedirs(f"cache/{args.data}")

    with Path(f"cache/{args.data}/tokenized_dataset.pkl").open("wb") as f:
        pickle.dump(tokenized_dataset, f)

def pre_def_clm(args, tokenizer, dataset): 
    def preprocess_func(examples):
        inputs = examples["document"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )  # is_split_into_words = True,
        labels = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        # labels[labels == tokenizer.pad_token_id] = -100 # ?
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_func,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset["train"].column_names,
    )
    if args.cache:
        cache_data(args, tokenized_dataset)
    return tokenized_dataset

def pre_concat(args, tokenizer, dataset):
    # Concat inputs and targets for CLM training!
    sep= "\nSummary: "
    dataset = dataset.map(
        lambda examples: {
            "labels": [
                examples["inputs_pretokenized"][i] + sep + examples["targets_pretokenized"][i].lstrip() for i in range(len(examples["targets_pretokenized"]))
            ]
        },
        # lambda examples : {'content' : [examples['inputs_pretokenized'][i] + " <se> " + examples['targets_pretokenized'][i].lstrip() for i in range(len(examples['targets_pretokenized']))]},
        batched=True,
        remove_columns=[
            "inputs",
            "targets",
            "targets_pretokenized",
        ],  # dataset["train"].column_names
        num_proc=args.num_proc,
    )

    def preprocess_func(examples):
        model_inputs = tokenizer(
            examples["inputs_pretokenized"],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            examples["labels"],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_func,
        batched=True,
        remove_columns=["inputs_pretokenized"],
        num_proc=args.num_proc,
    )
    if args.cached:
        cache_data(args, tokenized_dataset)
    return tokenized_dataset

def pre_seq2seq(args, tokenizer, dataset):
    # article + <sep> + summary form
    # sep 토큰 tokenizer 에 있는지 없는지 확인하기. -> 원래는 없는 듯.
    # tokenizer.add_special_tokens({'sep_token':'<sep>'}) # num_add_toks=1 이면 굳이 num_add_toks = tokenizer.~ 이렇게 안써도되지않나
    sep= "\nSummary: " 
    dataset = dataset.map(
        lambda examples: {
            "labels": [
                examples["article"][i] + sep + examples["one_sent_sum"][i].lstrip() # document, summary
                for i in range(len(examples["one_sent_sum"]))
            ]
        },
        batched=True,
        remove_columns=["one_sent_sum", "id"],
        num_proc=args.num_proc,
    )

    def preprocess_func(examples):
        model_inputs = tokenizer(
            examples["article"], # document
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            examples["labels"],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_func,
        batched=True,
        remove_columns=["article"], # document
        num_proc=args.num_proc,
    )
    if args.cached:
        cache_data(args, tokenized_dataset)
    return tokenized_dataset

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", default=10, type=int, help="training epoch")
    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=0.03, # 3e−3
        type=float,
        dest="lr",
        help="training learning rate",
    )  # constant lr 0.3??
    parser.add_argument(
        "--batch_size", "-bs", default=4, type=int, help="training batch size"
    )
    parser.add_argument(
        "--peft",
        default="prot",
        choices=["pret", "prot", "ia3"],
        help="which peft method to use - prefix tuning, prompt tuning, ia3",
    )
    parser.add_argument(
        "--vir_tok",
        default=30,
        type=int,
        help="prompt, prefix tuning num_virtual_token",
    )
    parser.add_argument(
        "--max_length", "-ml", default=1024, type=int, help="maximum sequence length"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-save_mode", type=str, default="best", choices=["all", "best"])
    parser.add_argument("--model_name_or_path", default="gpt2-large", help="base model") #  meta-llama/Llama-2-7b-chat-hf
    parser.add_argument(
        "--output_dir", default="output_pt", help="experiment result save directory"
    )
    parser.add_argument("--data_choice", default="gemini_new.json", choices=["p3", "org_xsum", "dataset.pkl", "gemini_new.json"])
    parser.add_argument(
        "--data_preprocess",
        default="seq2seq",
        choices=["def_clm", "concat", "seq2seq", "cached"],
        dest="data",
        help="data preprocess method for Causal LM",
    )
    parser.add_argument(
        "--cached", default=False, help="whether cache tokenized_dataset"
    )
    parser.add_argument(
        "--debug", default=False, help="data sampling with Subset for debugging"
    )
    parser.add_argument("--interval", default=17004, help="evaluate term")

    parser.add_argument(
        "--deepspeed",
        default="./deepspeed_config.json",
        help="deepspeed configuration (json) file directory location",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--num_proc",
        default=16,
        type=int,
        help="the number of subprocesses for mapping dataset",
    )  # CPU 는 32개는 할수 있다고 함..
    parser.add_argument(
        "--tokenized_dataset_cache",
        default=None,
        help="path to tokenized dataset cache",
    )  #'cache/seq2seq/tokenized_dataset.pkl', r'C:\Users\user\Downloads\concat\concat\tokenized_dataset'

    # Include DeepSpeed configuration arguments.
    # parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def main():
    args = add_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token="<pad>")# , pad_token="<pad>"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # For reproducibility
    if args.seed is not None:
        random.seed(args.seed)  # python random seed
        # np.random.seed(args.seed) # numpy random seed
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True  # add
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)  # add
        # torch.cuda.manual_seed_all(args.seed)
        # torch.set_deterministic(True)

    # 실험 결과 저장 directory
    if os.path.exists(args.output_dir):
        args.output_dir += datetime.today().strftime("_%Y%m%d_%H%M%S")
        print("   - Output directory is changed to avoid overlapping.")
        print("test")
    else:  # Trainer 에서는 mkdir 를 사용하지 않아도 자동으로 만들어준다
        pass

    if args.peft in ["pret", "prot", "ia3"]:
        peft_config = set_peft_config(args)
    else:
        pass

    if args.data_choice == 'p3':
        dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")

    elif args.data_choice == 'xsum':
        dataset = load_dataset("EdinburghNLP/xsum")

    elif args.data_choice == "dataset.pkl":
        with open(os.path.join("cache", args.data_choice), "rb") as f: 
            dataset = pickle.load(f)
    
    elif args.data_choice == "gemini_new.json":
        train_dataset = load_dataset('json', field='train', data_files='resum.json')['train']
        eval_dataset = load_dataset('json', field='test', data_files='resum.json',)['train']
        # with open('resum.json') as f:
        #     dataset = json.load(f)
        #     dataset = Dataset.from_dict(dataset) # 이러면 오류남.
    else:
        pass

    if args.data_choice == "gemini_new.json":
        train_dataset = pre_seq2seq(args, tokenizer, train_dataset)
        eval_dataset = pre_seq2seq(args, tokenizer, eval_dataset)
    else:
        if args.data == "def_clm":
            tokenized_dataset = pre_def_clm(args, tokenizer, dataset)

        elif args.data == "concat":  # p3 xsum dataset seq2seq
            tokenized_dataset = pre_concat(args, tokenizer, dataset)

        elif args.data == "seq2seq":  # original xsum dataset seq2seq
            tokenized_dataset = pre_seq2seq(args, tokenizer, dataset)

        elif args.data == "cached":
            with Path(args.tokenized_dataset_cache).open("rb") as f:
                tokenized_dataset = pickle.load(f)
        else:
            raise NotImplementedError
        
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["validation"]

    model.resize_token_embeddings(len(tokenizer))  # resize 는 반드시 get_peft_model(즉, peft를 씌우기전에) 해줘야한다!

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )  # tokenize 미리 한거를 input 으로 넣는데 여기 tokenizer 를 왜또넣지.ㅡㅡ

    def compute_metrics(eval_preds):
        return

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=1000,
        evaluation_strategy="epoch",
        logging_dir="log",
        logging_steps=args.interval,
        logging_first_step =True,
        save_steps=args.interval,
        seed=args.seed,
        prediction_loss_only=True,
        deepspeed=args.deepspeed,  # deepspeed 명령어는 run 할 때, train args 는 디버깅할 때 사용.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
        # optimizers = optimizer # lr_scheduler
        # neftune_noise_alpha=5,
    )
    trainer.train()


if __name__ == "__main__":
    main()
