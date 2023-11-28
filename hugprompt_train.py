from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Model, AutoModelForCausalLM, get_linear_schedule_with_warmup, default_data_collator
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType, PeftType, get_peft_config
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import torch
import os
import argparse
from datetime import datetime
import random

def evaluate(model, eval_loader, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels = batch['input_ids'])
            loss = outputs.loss
            eval_loss += loss.detach().float()

    eval_epoch_loss = eval_loss / len(eval_loader)
    eval_ppl = torch.exp(eval_epoch_loss) 
    return eval_epoch_loss, eval_ppl


def train(model, train_loader, eval_loader, optimizer, lr_scheduler, device, args):
    model = model.to(device)
    train_losses = []
    eval_losses = [1e5]
    tr_output_file = os.path.join(args.output_dir, 'tr_result.txt')
    te_output_file = os.path.join(args.output_dir, 'te_result.txt')
    with open(tr_output_file, 'w') as f:
        f.write('trloss, trppl\n')
    with open(te_output_file, 'w') as f:
        f.write('teloss, teppl\n')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels = batch['input_ids']) # , labels = batch['input_ids']
            loss = outputs.loss
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
            if step % args.interval == 17003:
                eval_epoch_loss, eval_ppl = evaluate(model, eval_loader, device)

                if args.save_mode == 'all':
                    model_name = 'model_ep_{ep:d}_loss_{ls:3.3f}.pt'.format(ep=epoch, ls=eval_epoch_loss)
                    model.save_pretrained(os.path.join(args.output_dir, model_name))
                elif args.save_mode == 'best':
                    model_name = 'model.pt'
                    if eval_epoch_loss < min(eval_losses):
                        model.save_pretrained(os.path.join(args.output_dir, model_name))
                        print('    - [Info] The checkpoint file has been updated.')
                else:
                    raise NotImplementedError
                
                eval_losses.append(eval_epoch_loss.item())
                with open(te_output_file, 'a') as f:
                    f.write(f'{eval_epoch_loss.item()}, {eval_ppl}\n')

        train_epoch_loss = train_loss / len(train_loader)
        train_ppl = torch.exp(train_epoch_loss)
        with open(tr_output_file, 'a') as f:
            f.write(f'{train_epoch_loss}, {train_ppl}\n')
        train_losses.append(train_epoch_loss.item())
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default=10, type=int,
                        dest='epochs', help='training epoch')
    parser.add_argument('--learning-rate', '-lr', default=0.3, type=float, 
                        dest='lr', help='training learning rate') # constant lr 0.3??
    parser.add_argument('--batch-size', '-bs', default=4, type=int,
                        dest='batch_size', help='training batch size')
    parser.add_argument('--vir_tok', default=30, type=int,
                        help = 'prompt, prefix tuning num_virtual_token')
    parser.add_argument('--max_length', '-ml', default=994, type=int, 
                        dest='max_length', help='maximum sequence length, 1024-args.vir_tok')
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--output_dir', default='output_pt',
                        help='experiment result save directory')
    
    parser.add_argument('--data_preprocess', default='seq2seq', choices = ['def_clm', 'concat', 'seq2seq'],
                        dest = 'data', help='data preprocess method for Causal LM')
    parser.add_argument('--debug', default=False, 
                        help='data sampling with Subset for debugging')
    parser.add_argument('--interval', default=17004,
                        help='evaluate term')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

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

    # 실험 결과 저장 directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        args.output_dir += datetime.today().strftime('_%Y%m%d_%H%M%S')
        os.makedirs(args.output_dir)
        print('   - Output directory is changed to avoid overlapping.')

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT, # prompt tuning embedding initial value type, 다른 종류는 RANDOM
        num_virtual_tokens=args.vir_tok, # output prompt size (아마)
        prompt_tuning_init_text="Summarize the following article with 1 sentence: ", # 프롬프트 초기화
        tokenizer_name_or_path=args.model_name_or_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              pad_token='<pad>') 
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("EdinburghNLP/xsum")

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
        a = 0

        dataset = dataset.map(
            lambda examples : {'content' : [examples['document'][i] +' <sep> '+ examples['summary'][i].lstrip() for i in range(len(examples['summary']))]},
            batched= True,
            remove_columns=['summary', 'id'],
            num_proc = 1)
        
        tokenized_dataset = dataset.map(
            lambda examples : tokenizer(examples['content'], padding='max_length', max_length=args.max_length, truncation=True, return_tensors="pt"),
            batched=True,
            num_proc = 1)

        # print(sum([1 for i in range(len(tokenized_dataset['train']['content'])) if 50257 in tokenized_dataset['train']['input_ids'][i]]))

        # def preprocess_func(examples):
        #     inputs = tokenizer(examples['document'])
        #     targets = tokenizer(examples['summary'])
        #     labels = inputs['input_ids'] + [tokenizer.sep_token_id] + targets['input_ids']
        #     text = tokenizer.encode(tokenizer.pad_token)*1022
        #     text[:len(labels)] = labels
        #     text = torch.tensor(text)
        #     model_inputs = {"inputs": torch.tensor(inputs), "labels": text}
        #     return model_inputs
        # 
        # tokenized_dataset = dataset.map(
        #     preprocess_func,
        #     batched=True,
        #     remove_columns=['summary']
        # )
        
    model.resize_token_embeddings(len(tokenizer)) 
    
    # For debugging
    if args.debug:
        num_train_idxs = list(range(4))
        train_dataset = Subset(tokenized_dataset['train'], num_train_idxs)
        eval_dataset = Subset(tokenized_dataset['validation'], num_train_idxs)
        print('done')
    
    else:
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0, # 8e4,
        num_training_steps = (len(train_loader)*args.epochs)
    )

    train(model, train_loader, eval_loader, optimizer, lr_scheduler, device, args)



if __name__ == '__main__':
    main()