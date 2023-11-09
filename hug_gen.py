from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

# import nltk.translate.bleu_score as bleu
from tqdm import tqdm
import os
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np

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
    epoch_step = list(np.arange((1/3), (31/3), (1/3)))

    fig, ax = plt.subplots(2, 2)
    # Train Loss
    ax[0, 0].plot(epoch, trloss, color = 'b', marker='o', markersize=3)
    ax[0, 0].set_title('Train Loss')
    ax[0, 0].set_xlabel('epochs')
    ax[0, 0].set_ylabel('loss')
    # ax[0, 0].set_ylim([0.9, 1.2])
    ax[0, 0].grid(True)

    # Train ppl
    ax[0, 1].plot(epoch, trppl, color = 'g', marker='o', markersize=3)
    ax[0, 1].set_title('Train PPL')
    ax[0, 1].set_ylabel('ppl')
    ax[0, 1].grid(True)

    # Eval Loss
    ax[1, 0].plot(epoch_step, teloss, color = 'b', marker='o', markersize=3)
    ax[1, 0].set_title('Eval Loss')
    ax[1, 0].set_xticks(list(range(0, 11)))
    ax[1, 0].grid(True)

    # Eval ppl
    ax[1, 1].plot(epoch_step, teppl, color = 'g', marker='o', markersize=3)
    ax[1, 1].set_title('Eval PPL')
    ax[1, 1].set_xticks(list(range(0, 11)))
    ax[1, 1].grid(True)

    fig.tight_layout()
    plt.show()
    plt.savefig('graph')


def test_gen(model, dataset, tokenizer, device, args):
    model = model.to(device)
    test_loader = DataLoader(dataset, pin_memory=True)

    gen_dir = os.path.join(args.output_dir, 'generate.txt')
    with open(gen_dir, 'w') as f:
        f.write(f'<Generated Output>\n\n')

    for step, inputs in enumerate(tqdm(test_loader)):
        input_ids = tokenizer.encode(inputs['inputs_pretokenized'][0], return_tensors='pt')
        # outputs = model.generate(input_ids=input_ids.to(device), max_new_tokens=20, return_full_text=False)
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
    print('Done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--output_dir', default='output',
                        help='experiment result save directory')  # C:/Users/mari970/Downloads/output_20231019_090057/
    parser.add_argument('--max_length', '-ml', default=1004, type=int, 
                        dest='max_length', help='maximum sequence length')
    parser.add_argument('--met_choice', choice=['peft', 'original'], default='peft')
    parser.add_argument('--data_choice', choices=['p3', 'org_xsum'], default='')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    if args.met_choice == 'original':
        # 일반 AutoModel 로는 PEFT 로 학습한 모델을 불러올 수 없다. (당연함. Adapter 같은애들은 새로운 모듈을 추가.)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path) 

    elif args.met_choice == 'peft':
        model_chkpt = os.path.join(args.output_dir, 'model.pt')
        model = AutoPeftModelForCausalLM.from_pretrained(model_chkpt)

    else:
        print('ERROR')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token='<pad>')

    if args.data_choice == 'p3':
        # Prefix tuning : p3-xsum 데이터셋 (사실 이거말고 일반 xsum 데이터셋을 쓰는게 맞지만 이렇게 학습시켜버려서,,)
        dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")['test']
        dataset = dataset.remove_columns(['inputs', 'targets'])

    elif args.data_choice == 'org_xsum':
        # Prompt tuning 
        dataset = load_dataset("EdinburghNLP/xsum")['test']
        dataset = dataset.remove_columns(['inputs', 'targets'])
        
    else:
        print('ERROR')
    
    
    test_gen(model, dataset, tokenizer, device, args)

    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # draw_graph(args.output_dir) 


if __name__ == '__main__':
    main()