# from datasets import load_dataset
# dataset = load_dataset("xsum")

# Dataset unzip
# import tarfile
# file = tarfile.open('XSUM-EMNLP18-Summary-Data-Original.tar.gz')
# file.extractall('./XSUMdata')
# file.close()

# Prompting
import os
file_path = 'XSUMdata/bbc-summary-data/'
file_list = []
with open(file_name, 'r') as f: