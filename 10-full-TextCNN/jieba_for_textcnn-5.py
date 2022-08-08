import argparse
from tqdm import tqdm
import jieba

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file


print('spliting')
lengths = []#每个句子的长度
with open(input_file, 'r', encoding='utf-8') as f:
    preprocessed_data = []
    for line in tqdm(f):
        line = line.strip()
        if not line:
            continue
        content, label = line.split('\t')
        splited_content = jieba.lcut(content)
        lengths.append(len(splited_content))#统计句子长度

        content_text = ' '.join(splited_content)
        text = '\t'.join([content_text, label])
        preprocessed_data.append(text)
    
print('saving')
with open(output_file, 'w', encoding='utf-8') as f:
    for line in preprocessed_data:
        f.write(line + '\n')
#pad_size





import matplotlib.pyplot as plt

plt.hist(lengths, bins=10)
plt.show()