import os
from datasets import load_dataset
current_path = os.path.dirname(__file__)
ChnSentiCorp_path = os.path.join(current_path, 'ChnSentiCorp')
# print(ChnSentiCorp_path);exit()
dataset = load_dataset(path='ChnSentiCorp', split='train')#seamew/ChnSentiCorp
# dataset = load_dataset(path=ChnSentiCorp_path, split='train')
print(dataset)
print(dataset[0])
