from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
import torch

# import nltk.translate.bleu_score as bleu
from tqdm import tqdm
import os
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

def draw_graph(dir):
    tr = os.path.join(dir, 'tr_result.txt')
    te = os.path.join(dir, 'te_result.txt')
    trloss, trppl, teloss, teppl = [], [], [], []
    with open(tr, 'r') as f:
        l = f.readlines()
        for line in l[1:]:
            a, b = line.split(',')
            trloss.append(float(a))
            trppl.append(float(b.strip()))

    with open(te, 'r') as f:
        l = f.readlines()
        for line in l[1:]:
            a, b = line.split(',')
            teloss.append(float(a))
            teppl.append(float(b))

    epoch = list(range(1, 11, 1))
    epoch_step = list(range(1, 31, 1))

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(epoch, trloss, color = 'b', marker='o', markersize=3)
    ax[0, 0].set_title('Train Loss')
    ax[0, 0].set_xlabel('epochs')
    ax[0, 0].set_ylabel('loss')
    # ax[0, 0].set_ylim([0.9, 1.2])
    ax[0, 0].grid(True)

    ax[0, 1].plot(epoch, trppl, color = 'g', marker='o', markersize=3)
    ax[0, 1].set_title('Train PPL')
    ax[0, 1].grid(True)

    ax[1, 0].plot(epoch_step, teloss, color = 'b', marker='o', markersize=3)
    ax[1, 0].set_title('Eval Loss')
    # ax[1, 0].set_xticks([0.9, 1.0])
    ax[1, 0].set_xticks([1, 10])
    ax[1, 0].grid(True)

    ax[1, 1].plot(epoch_step, teppl, color = 'g', marker='o', markersize=3)
    ax[1, 1].set_title('Eval PPL')
    ax[1, 1].grid(True)

    fig.tight_layout()
    plt.show()
    plt.savefig('graph')

def save_txt(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    gen_dir = os.path.join(args.output_dir, 'generate.txt')
    with open(gen_dir, 'w') as f:
        f.write(f'<Generated Output>\n\n')

    with open(gen_dir, 'a',encoding='UTF-8') as f:
        f.write(f"[{step}th Generated Output]\n")
        inp = inputs['document'][0]
        f.write(f"{inp}\n\n")
        tar = inputs['summary'][0]
        f.write(f"{tar}\n\n")
        f.write(f"{tokenizer.decode(outputs[0,:].tolist())}\n\n")
        f.write('\n')
    print("  - Save generated outputs with text file .txt")

def save_json(args, out_dict_list):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    gen_dir = os.path.join(args.output_dir, 'generate.json')
    with open(gen_dir, 'w') as f:
        json.dump(out_dict_list, f)

    print("  - Save generated outputs with json file .json")

"""
The index of original and instructed(p3) xsum dataset not same for prompt, prefix, ia3 PEFT.
To solve, add instruction 'Summarize this document:' to original xsum dataset.
Prompt and Prefix tuning dont need text instruction(prompt) they use instead tensor prompt!
"""
def make_instructed(dataset, args):
    if args.data_choice == 'p3':
        dataset = dataset.remove_columns(['inputs', 'targets'])
        print('  - Using dataset p3 xsum instructed dataset')

    elif args.data_choice == 'xsum':
        inst= "Summarize this document: "
        sep= "\nSummary: "

        dataset = dataset.map(
            lambda examples: {
                "labels": [
                    inst + examples["document"][i] + sep for i in range(len(examples["summary"]))
            ]},
            batched=True,
            num_proc=args.num_proc,
            remove_columns=['id']
        )

        print("  - Making original xsum dataset same with P3 xsum by adding 'Summarize this document:' instruction, 'Summary: ' separate token(?)")
    return dataset

def set_single_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token='<pad>')

    if args.met_choice == 'original':
        # 일반 AutoModel 로는 PEFT 로 학습한 모델을 불러올 수 없다. (당연함. Adapter 같은애들은 새로운 모듈을 추가.)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        print('original trained model')

    elif args.met_choice == 'peft':
        model_chkpt = os.path.join(args.output_dir, 'model.pt')
        model = AutoPeftModelForCausalLM.from_pretrained(model_chkpt)
        print(f'peft trained model {args.output_dir}')

    else:
        raise NotImplementedError
    
    if args.data_choice == 'p3':
        dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")['test']
    elif args.data_choice == 'xsum':
        dataset = load_dataset('EdinburghNLP/xsum')['test']

    idx_list = list(range(0, len(dataset)))
    num_train_idxs = random.sample(idx_list, 100)
    dataset = Subset(dataset, num_train_idxs)
    print('  - Dataset sampling randomly for generation')

    return dataset, model, tokenizer


def set_multi(args):
    dataset = load_dataset('EdinburghNLP/xsum')['test']
    # IA3 는 instruct 를 추가해줘야한다.
    dataset2 = make_instructed(dataset, args)

    idx_list = list(range(0, len(dataset)))
    num_train_idxs = random.sample(idx_list, 100)
    dataset = Subset(dataset, num_train_idxs)
    print('  - Dataset sampling randomly for generation')

    return dataset, dataset2

def test_single(model, dataset, tokenizer, device, args):
    model = model.to(device)
    test_loader = DataLoader(dataset, pin_memory=True)
    gen_dir = os.path.join(args.output_dir, 'generate.txt')
    with open(gen_dir, 'w') as f:
        f.write(f'<Generated Output>\n\n')
    if args.data_choice == 'p3':
        for step, inputs in enumerate(tqdm(test_loader)):
            input_ids = tokenizer.encode(inputs['inputs_pretokenized'][0], return_tensors='pt')
            outputs = model.generate(input_ids=input_ids.to(device),
                                     max_length=512,
                                     do_sample=True,
                                     repetition_penalty=0.5,
                                     eos_token_id= tokenizer.eos_token_id,
                                     bos_token_id=tokenizer.bos_token_id,
                                     use_cache=True
                                     )
            if step % 1000 == 0:
                print(step)
                with open(gen_dir, 'a',encoding='UTF-8') as f:
                    f.write(f"[{step}th Generated Output]\n")
                    inp = inputs['inputs_pretokenized'][0]# .encode('utf8')
                    f.write(f"{inp}\n\n")
                    tar = inputs['targets_pretokenized'][0]# .encode('utf8')
                    f.write(f"{tar}\n\n")
                    f.write(f"{tokenizer.decode(outputs[0,:].tolist())}\n\n")
                    f.write('\n')

    elif args.data_choice == 'xsum':
        for step, inputs in enumerate(tqdm(test_loader)):
            input_ids = tokenizer.encode(inputs['document'][0], return_tensors='pt')
            outputs = model.generate(input_ids=input_ids.to(device), max_new_tokens=10) # return_full_text=False
            if step % 1000 == 0:
                print(step)
                with open(gen_dir, 'a',encoding='UTF-8') as f:
                    f.write(f"[{step}th Generated Output]\n")
                    inp = inputs['document'][0]
                    f.write(f"{inp}\n\n")
                    tar = inputs['summary'][0]
                    f.write(f"{tar}\n\n")
                    f.write(f"{tokenizer.decode(outputs[0,:].tolist())}\n\n")
                    f.write('\n')
        print('Done')

def test_multi(dataset, dataset2, device, args):
    xsum_loader = DataLoader(dataset, pin_memory=True)
    p3_loader = DataLoader(dataset2, pin_memory=True)
    out_dict_list = []
    
    # 3개 모델에 대해서 generate 를 돌려야 하는데 3번 for 문을 돌리는 것이 효율적인지 아니면 한번에 2개 모델에 넣어서 생성을 만드는 것이 나은지
    # -> 생각하기 귀찮으니까 그냥 따로따로 하자..

    # Prefix tuning model
    model_chkpt = os.path.join('pret_concat/output_20231019_090057', 'model.pt')
    model = AutoPeftModelForCausalLM.from_pretrained(model_chkpt)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token='<pad>')

    for step, inputs in enumerate(tqdm(xsum_loader)):
        input_ids = tokenizer.encode(inputs['document'][0], truncation=True, return_tensors='pt') 
        outputs = model.generate(input_ids=input_ids.to(device),
                                    max_length=512,
                                    do_sample=True,
                                    repetition_penalty=0.5,
                                    eos_token_id= tokenizer.eos_token_id,
                                    bos_token_id=tokenizer.bos_token_id,
                                    use_cache=True
                                    )
        out_dict = {'Article':inputs['document'][0], 'target': inputs['summary'][0], 'pret_out': tokenizer.decode(outputs[0,:].tolist())}
        out_dict_list.append(out_dict)

    # Prompt tuning model
    model_chkpt = os.path.join('prot_concat/output_pt', 'model.pt')
    model = AutoPeftModelForCausalLM.from_pretrained(model_chkpt)
    model.to(device)

    for step, inputs in enumerate(tqdm(xsum_loader)):
        input_ids = tokenizer.encode(inputs['document'][0], truncation=True, return_tensors='pt')
        outputs = model.generate(input_ids=input_ids.to(device),
                                    max_length=512,
                                    do_sample=True,
                                    repetition_penalty=0.5,
                                    eos_token_id= tokenizer.eos_token_id,
                                    bos_token_id=tokenizer.bos_token_id,
                                    use_cache=True
                                    )
        out_dict_list[step]['prot_out'] = tokenizer.decode(outputs[0,:].tolist())

    # (IA)3 model
    model_chkpt = 'ia3_concat/LLM_PEFT/output_pt_20231210_163501_ia3/checkpoint-119028'
    model = AutoPeftModelForCausalLM.from_pretrained(model_chkpt)
    model.to(device)

    for step, inputs in enumerate(tqdm(p3_loader)):
        input_ids = tokenizer.encode(inputs['document'][0], truncation=True, return_tensors='pt')
        outputs = model.generate(input_ids=input_ids.to(device),
                                    max_length=512,
                                    do_sample=True,
                                    repetition_penalty=0.5,
                                    eos_token_id= tokenizer.eos_token_id,
                                    bos_token_id=tokenizer.bos_token_id,
                                    use_cache=True
                                    )

        out_dict_list[step]['ia3_out'] = tokenizer.decode(outputs[0,:].tolist())

    save_json(args, out_dict_list)
    print('  - Generation Done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--output_dir', default='./generated_output',
                        help='experiment result save directory')  #  output_pt_20231102_004015 , output_20231019_090057
    parser.add_argument('--max_length', '-ml', default=984, type=int,
                        dest='max_length', help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--met_choice', default='multi', choices=['peft', 'original', 'multi'])
    parser.add_argument('--data_choice', default='xsum', choices=['p3', 'xsum'])
    parser.add_argument('--debug', default=True)
    parser.add_argument('--num_proc', default=16, help='The number of process of mapping preprocess')
    args = parser.parse_args()

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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print('Device: ', device)

    if args.met_choice == 'multi':
        dataset, dataset2 = set_multi(args)
        test_multi(dataset, dataset2, device,args)

    else:
        dataset, model, tokenizer = set_single_model(args)
        test_single(model, dataset, tokenizer, device, args)
        

    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # draw_graph(args.output_dir)
    
if __name__ == '__main__':
    main()