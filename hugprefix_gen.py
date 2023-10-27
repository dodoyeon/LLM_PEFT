from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

import nltk.translate.bleu_score as bleu
from tqdm import tqdm
import os
import random
import argparse
import matplotlib.pyplot as plt

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


def test_gen(model, dataset, tokenizer, args):
    test_loader = DataLoader(dataset, pin_memory=True)
    for step, inputs in enumerate(test_loader):
        input_ids = tokenizer.encode(inputs['inputs_pretokenized'][0], return_tensors='pt')
        outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
        if step % 1000 == 0:
            print(f'[{step}th Generated Output]')
            print(inputs['inputs_pretokenized'][0])
            print(inputs['targets_pretokenized'][0])
            print(tokenizer.decode(outputs[0,:].tolist()))
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--output_dir', default='C:/Users/mari970/Downloads/output_20231019_090057/output_20231019_090057',
                        help='experiment result save directory')  # 'output_20231019_102420',
    # parser.add_argument('--data_preprocess', default='def_clm', choices = ['def_clm', 'concat'],
    #                     dest = 'data', help='data preprocess method for Causal LM')
    parser.add_argument('--max_length', '-ml', default=1004, type=int, # 1024 인데 prefix length=20 이라서,
                        dest='max_length', help='maximum sequence length')
    # parser.add_argument()
    args = parser.parse_args()

    model_chkpt = os.path.join(args.output_dir, 'model.pt')
    # model = AutoModelForCausalLM.from_pretrained(model_chkpt) # 일반 AutoModel 로는 PEFT 로 학습한 모델을 불러올 수 없다. (당연함. Adapter 같은애들은 새로운 모듈을 추가.)
    model = AutoPeftModelForCausalLM.from_pretrained(model_chkpt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token='<pad>')

    dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")['test']
    # dataset_te = DatasetDict({test: dataset['test']})
            
    dataset = dataset.remove_columns(['inputs', 'targets'])
    
    # test_gen(model, dataset, tokenizer, args)

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    draw_graph(args.output_dir)


if __name__ == '__main__':
    main()