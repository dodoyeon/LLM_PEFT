# from datasets import load_dataset
# dataset = load_dataset("xsum")

import tarfile

file = tarfile.open('XSUM-EMNLP18-Summary-Data-Original.tar.gz')
file.extractall('./XSUMdata')
file.close()