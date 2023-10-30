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


def test_gen(model, dataset, tokenizer, args):
    gen_dir = os.path.join(args.output_dir, 'generate.txt')
    with open(gen_dir, 'w') as f:
        f.write(f'<Generated Output>\n\n')
    test_loader = DataLoader(dataset, pin_memory=True)
    for step, inputs in enumerate(tqdm(test_loader)):
        input_ids = tokenizer.encode(inputs['inputs_pretokenized'][0], return_tensors='pt')
        outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
        if step % 1000 == 0:
            print(step)
            with open(gen_dir, 'a') as f:
                f.write(f'[{step}th Generated Output]\n')
                f.write(f'{inputs["inputs_pretokenized"][0]}\n\n')
                f.write(f'{inputs["targets_pretokenized"][0]}\n\n')
                f.write(f'{tokenizer.decode(outputs[0,:].tolist())}\n\n')
                f.write('\n')
    print('Done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--output_dir', default='output_20231019_090057',
                        help='experiment result save directory')  # C:/Users/mari970/Downloads/output_20231019_090057/
    parser.add_argument('--max_length', '-ml', default=1004, type=int, 
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
    
    test_gen(model, dataset, tokenizer, args)

    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # draw_graph(args.output_dir) 


if __name__ == '__main__':
    main()