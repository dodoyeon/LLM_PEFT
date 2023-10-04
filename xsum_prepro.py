from datasets import get_dataset_split_names, load_dataset
# dataset = load_dataset("xsum")

# Dataset unzip
# import tarfile
# file = tarfile.open('XSUM-EMNLP18-Summary-Data-Original.tar.gz')
# file.extractall('./XSUMdata')

# P3 dataset uses XSum dataset
'''
[config_name] = 
'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_DOC_given_above_write_one_sentence', 'xsum_DOC_how_would_you_rephrase_few_words', 'xsum_DOC_tldr', 
'xsum_DOC_write_summary_of_above', 'xsum_article_DOC_summary', 'xsum_college_roommate_asked_DOC_so_I_recap', 'xsum_read_below_DOC_write_abstract', 
'xsum_summarize_DOC', 'xsum_summarize_this_DOC_summary',
'''
from datasets import get_dataset_split_names
# get_dataset_split_names(path="bigscience/P3", config_name='xsum_summarize_this_DOC_summary') 
dataset = load_dataset("bigscience/P3", name='xsum_article_DOC_summary', split="train")
print('done')

# Prompting -> 원하던 프롬프트 형식은 아니지만 P3 에서 XSUM 프롬프팅이 되어있기때문에 이걸 먼저 사용해본다.
# import os
# file_path = 'XSUMdata/bbc-summary-data/'
# file_list = [os.listdir(file_path)]

# for file_name in file_list:
#     with open(file_name, 'w') as f:
#         # add 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' 
#         # '### Instruction:\n' '### Response:\n'
#         print('done')