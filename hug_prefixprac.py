from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Model, AutoModelForCausalLM, get_linear_schedule_with_warmup, default_data_collator
from peft import PrefixTuningConfig, get_peft_model, TaskType # 따로 peft install 해야함
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
import argparse

def evaluate(model, eval_loader, device, args):
    model.eval()
    eval_loss = 0
    for step, batch in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            outputs = model(batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()

    eval_epoch_loss = eval_loss / len(eval_loader)
    eval_ppl = torch.exp(eval_epoch_loss)
    return eval_epoch_loss, eval_ppl



def train(model, train_loader, eval_loader, optimizer, lr_scheduler, device, args):
    model = model.to(device)
    train_losses = []
    eval_losses = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels = batch['input_ids'])
            loss = outputs.loss
            train_loss += loss.detach().float()# .item()?
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        eval_epoch_loss, eval_ppl = evaluate(model, eval_loader, device, args)

        train_epoch_loss = train_loss / len(train_loader)
        train_ppl = torch.exp(train_epoch_loss)
        train_losses.append(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        if args.save_mode == 'all':
            model_name = 'model_ep_{ep:d}_accu_{accu:3.3f}.pt'.format(ep=epoch, accu=100*eval_epoch_loss)
            model.save_pretrained(os.path.join(args.result_dir, model_name))
        elif args.save_mode == 'best':
            model_name = 'model.pt'
            if eval_epoch_loss < min(eval_losses):
                model.save_pretrained(os.path.join(args.result_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')
        else:
            raise NotImplementedError
        
        eval_losses.append(eval_epoch_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default=20, type=int,
                        dest='epochs', help='training epoch')
    parser.add_argument('--learning-rate', '-lr', default=1e-2, type=float,
                        dest='lr', help='training learning rate')
    parser.add_argument('--batch-size', '-bs', default=8, type=int,
                        dest='batch_size', help='training batch size')
    parser.add_argument('--max_length', '-ml', default=1004, type=int, # 1024 인데 prefix length=20 이라서,
                        dest='max_length', help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=None) # 허깅페이스 사용하면 굳이 seed 고정할 필요가 없나??
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--result_dir', default='output',
                        dest = 'result_dir', help='experiment result save directory')
    
    parser.add_argument('--data_preprocess', default='concat', choices = ['def_clm', 'concat'],
                        dest = 'data', help='data preprocess method for Causal LM')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM, # TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        num_virtual_tokens=20
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              pad_token='<pad>') # -> 이걸하면 vocab size 가 커져서 Index out of range 문제가 뜬다.
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.resize_token_embeddings(len(tokenizer)) # 위 주석의 문제를 해결하기 위해 이렇게 세팅한다.

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")
    # dataset = dataset["train"].train_test_split(test_size=0.1)
    # dataset["validation"] = dataset["test"]
    # del dataset["test"]
    print('done')

    if args.data == 'def_clm':
        def preprocess_func(examples):
            inputs = examples['inputs_pretokenized']
            targets = examples['targets_pretokenized']
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
        lambda x : {'sent_forclm' : [x['inputs_pretokenized'][i] + x['targets_pretokenized'][i].lstrip() for i in range(len(x['targets_pretokenized']))]},
        batched= True,
        remove_columns=dataset["train"].column_names,
        num_proc = 1)

        tokenized_dataset = dataset.map(
            lambda examples : tokenizer(examples['sent_forclm'], padding='max_length', max_length=args.max_length, truncation=True, return_tensors="pt"),
            batched=True,
            num_proc = 1
            )

    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['validation']

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 8e4,
        num_training_steps = (len(train_loader)*args.epochs)
    )

    train(model, train_loader, eval_loader, optimizer, lr_scheduler, device, args)
    


if __name__ == '__main__':
    main()