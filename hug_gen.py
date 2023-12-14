from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader, Subset
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
    plt.savefig('graph')


def test_gen(model, dataset, tokenizer, device, args):
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


def use_instructed():
    dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")['test']
    dataset = dataset.remove_columns(['inputs', 'targets'])
    print('  -Dataset p3 xsum instructed dataset')
    return dataset


"""
The index of original and instructed(p3) xsum dataset not same for prompt, prefix, ia3 PEFT.
To solve, add instruction 'Summarize this document:' to original xsum dataset.
Prompt and Prefix tuning dont need text instruction(prompt) they use instead tensor prompt!
"""

def make_instructed(args):
    if args.data_choice == 'p3':
        pass

    elif args.data_choice == 'xsum':
        def preprocess_func(examples):
            sep1= "Summarize this document: "
            sep2= "\nSummary: "
            examples[]

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=['id'] 
        )

    print("  -Making original xsum dataset same with P3 xsum by adding 'Summarize this document:' instruction, 'Summary: ' separate token(?)")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'gpt2-large',
                        dest ='model_name_or_path', help='base model')
    parser.add_argument('--output_dir', default='.',
                        help='experiment result save directory')  #  output_pt_20231102_004015 , output_20231019_090057
    parser.add_argument('--max_length', '-ml', default=984, type=int,
                        dest='max_length', help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--met_choice', default='original', choices=['peft', 'original'])
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Device: ', device)

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
    
    dataset = load_dataset("bigscience/P3", name="xsum_summarize_this_DOC_summary")['test']
    idx_list = list(range(0, len(dataset)))
    num_train_idxs = random.sample(idx_list, 100)

    dataset = Subset(dataset, num_train_idxs)
    print('  - Dataset sampling randomly for generation')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, pad_token='<pad>')
    
    # Prefix tuning : p3-xsum 데이터셋 (사실 이거말고 일반 xsum 데이터셋을 쓰는게 맞지만 이렇게 학습시켜버려서,,)
    # use_instructed()
    dataset = make_instructed()
    
    if args.debug:
        debug_gen(model, dataset, tokenizer, device, args)

    else:
        # for evaluation metric like ROUGE, etc. [Not Implemented]
        test_gen(model, dataset, tokenizer, device, args)

    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # draw_graph(args.output_dir)
    
if __name__ == '__main__':
    main()