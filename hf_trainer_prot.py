from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, TrainingArguments
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType, PeftType, get_peft_config
import torch
# import deepspeed

import argparse
import random
import os

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default=10, type=int,
                         help='training epoch')
    parser.add_argument('--learning_rate', '-lr', default=0.3, type=float, 
                        dest='lr', help='training learning rate') # constant lr 0.3??
    parser.add_argument('--batch_size', '-bs', default=4, type=int,
                         help='training batch size')
    parser.add_argument('--vir_tok', default=30, type=int,
                        help = 'prompt, prefix tuning num_virtual_token')
    parser.add_argument('--max_length', '-ml', default=994, type=int, 
                         help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                         help='base model')
    parser.add_argument('--output_dir', default='output_pt',
                        help='experiment result save directory')
    
    parser.add_argument('--data_preprocess', default='seq2seq', choices = ['def_clm', 'concat', 'seq2seq'],
                         dest = 'data', help='data preprocess method for Causal LM')
    parser.add_argument('--debug', default=False, 
                        help='data sampling with Subset for debugging')
    parser.add_argument('--interval', default=17004,
                        help='evaluate term')
    parser.add_argument('--deepspeed_use', default=True,
                       help='whether use deepspeed lib or not')
    
    # Include DeepSpeed configuration arguments.
    # parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

def main():
    args = add_arguments()
    dataset = load_dataset("EdinburghNLP/xsum")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              pad_token='<pad>')
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT, # prompt tuning embedding initial value type, 다른 종류는 RANDOM
        num_virtual_tokens=args.vir_tok, # output prompt size (아마)
        prompt_tuning_init_text="Summarize the following article with 1 sentence: ", # 프롬프트 초기화
        tokenizer_name_or_path=args.model_name_or_path
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    if args.data == 'def_clm':
        def preprocess_func(examples):
            inputs = examples['document']
            targets = examples['summary']
            model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length= args.max_length, return_tensors='pt') # is_split_into_words = True,
            labels = tokenizer(targets, padding='max_length', truncation=True, max_length= args.max_length, return_tensors='pt')
            labels = labels["input_ids"]
            # labels[labels == tokenizer.pad_token_id] = -100 # ?
            model_inputs["labels"] = labels
            return model_inputs


        tokenized_dataset = dataset.map(
            preprocess_func,
            batched=True,
            num_proc = 1,
            remove_columns=dataset["train"].column_names
            )
        
    elif args.data == 'concat':
        # Concat inputs and targets for CLM training
        dataset = dataset.map(
            lambda examples : {'content' : [examples['document'][i] + examples['summary'][i].lstrip() for i in range(len(examples['summary']))]},
            # lambda examples : {'content' : [examples['inputs_pretokenized'][i] + examples['targets_pretokenized'][i].lstrip() for i in range(len(examples['targets_pretokenized']))]},
            batched= True,
            remove_columns=dataset["train"].column_names,
            num_proc = 1)

        tokenized_dataset = dataset.map(
            lambda examples : tokenizer(examples['content'], padding='max_length', max_length=args.max_length, truncation=True, return_tensors="pt"),
            batched=True,
            num_proc = 1
            )  

    elif args.data == 'seq2seq':
        # article + <sep> + summary form
        # sep 토큰 tokenizer 에 있는지 없는지 확인하기. -> 원래는 없는 듯.
        tokenizer.add_special_tokens({'sep_token':'<sep>'}) # num_add_toks=1 이면 굳이 num_add_toks = tokenizer.~ 이렇게 안써도되지않나

        dataset = dataset.map(
            lambda examples : {'content' : [examples['document'][i] +' <sep> '+ examples['summary'][i].lstrip() for i in range(len(examples['summary']))]},
            batched= True,
            remove_columns=['summary', 'id'],
            num_proc = 1)
        
        tokenized_dataset = dataset.map(
            lambda examples : tokenizer(examples['content'], padding='max_length', max_length=args.max_length, truncation=True, return_tensors="pt"),
            batched=True,
            num_proc = 1)
        
    model.resize_token_embeddings(len(tokenizer)) 

    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['validation']

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0, # 8e4,
        num_training_steps = ((dataset['train'].shape[0]//args.batch_size+1)*args.epochs) # trainloader 사용하지 않기 때문에
    )

    # For reproducibility
    if args.seed is not None:
        random.seed(args.seed) # python random seed
        # np.random.seed(args.seed) # numpy random seed
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True # add
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed) # add
        # torch.cuda.manual_seed_all(args.seed)  # add
        # torch.set_deterministic(True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_dir='log.txt',
        logging_strategy="epoch",
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        seed=args.seed,
        deepspeed=True
        
    )

    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        dataset_batch_size = args.batch_size,
        tokenizer = tokenizer,
        dataset_text_field="text",
        optimizers = lr_scheduler,
        max_seq_length=1024,
        peft_config = peft_config,
        # neftune_noise_alpha=5,
    )
    trainer.train()

if __name__ == "__main__":
    main()