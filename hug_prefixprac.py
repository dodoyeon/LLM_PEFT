from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Model, get_linear_schedule_with_warmup, default_data_collator
from peft import PrefixTuningConfig, get_peft_model, TaskType
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
    model.train()
    train_losses = []
    eval_losses = [1e5]
    for epoch in range(args.epochs):
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
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
    parser.add_argument('--max_length', '-ml', default=1024, type=int,
                        dest='max_length', help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=None) # 허깅페이스 사용하면 굳이 seed 고정할 필요가 없나??
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--result_dir', default='output',
                        dest = 'result_dir', help='experiment result save directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM, # TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        num_virtual_tokens=20
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              pad_token='<pad>')
    model = GPT2Model.from_pretrained(args.model_name_or_path)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")
    # dataset = dataset["train"].train_test_split(test_size=0.1)
    # dataset["validation"] = dataset["test"]
    # del dataset["test"]
    print('done')

    # dataset = dataset.map(
    #     # Concat inputs and targets for CLM training
    #     lambda x : {'sent_forclm' : [x['inputs_pretokenized'][i] + x['targets_pretokenized'][i].lstrip() for i in range(len(x['targets_pretokenized']))]},
    #     batched= True,
    #     remove_columns=['inputs', 'targets'],
    #     num_proc = 1
    # )
    
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['inputs_pretokenized'], padding='max_length', truncation=True, max_length= args.max_length, return_tensors='pt'), # is_split_into_words = True,
        lambda examples: tokenizer(examples['targets_pretokenized'], padding='max_length', truncation=True, max_length= args.max_length, return_tensors='pt'),
        batched=True, # sent_forclm
        num_proc = 1,
        remove_columns=dataset["train"].column_names
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