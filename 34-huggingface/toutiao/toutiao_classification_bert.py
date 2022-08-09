from pickletools import optimize
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
data_path = '/Users/m1pro/Desktop/file/datasets/中文数据集/toutiao-text-classfication-dataset-master/toutiao_cat_data.txt'

class ToutiaoDataset(data.Dataset):
    def __init__(self, data_path) -> None:
        super(ToutiaoDataset, self).__init__()
        self.build(data_path)

    def build(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = []
            labels = []
            for line in f:
                    _, category, _, text, key_word = line.strip().split('_!_')

                    labels.append(int(category)-100)
                    texts.append(text)
                    # print(texts[-1], labels[-1]);input()
            self.texts = texts
            self.labels = labels


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text_i = self.texts[i]
        label_i = self.labels[i]
        return text_i, label_i

toutiao_dataset = ToutiaoDataset(data_path)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
def collate_fn(text_label):
    #text_label =[
    # ('我是大肥猪', 0), 
    # ('我是小肥猪', 1),
    # ('兔子好可爱', 2)
    # ]
    texts = [text for text, _ in text_label]
    labels = [label for _, label in text_label]
    max_len = min(max(len(t) for t in texts) + 2, 512)

    # texts = [
    #     '我是大肥猪',
    #     '我是小肥猪',
    #     '兔子好可爱'
    # ]
    data = bert_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        add_special_tokens=True,

        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt',#tf,pt,np

        return_token_type_ids=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    # for key, value in data.items():
    #     print(key, ':', value);input()

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


#(batch_size, seq_len, hidden_size)
pretrained = BertModel.from_pretrained('bert-base-chinese')
# pretrained = BertModel()
#pretrained.load_state_dict(xx)
class ToutiaoClassification(nn.Module):
    def __init__(self, hidden_size, num_classes) -> None:
        super(ToutiaoClassification, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
        #fc(.shape = (batch_size, hidden_size))
        prediction = self.fc(out.last_hidden_state[:, 0, :])
        return prediction

lr = 5e-4
batch_size = 32
shuffle = True
epochs = 4

dataloader = data.DataLoader(toutiao_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
model = ToutiaoClassification(hidden_size=768, num_classes=17)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(epochs):
    counter = 0
    for input_ids, attention_mask, token_type_ids, label in dataloader:
        prediciton = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
        
        loss = criterion(prediciton, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediciton = prediciton.argmax(dim=1)
        accuracy = (prediciton == label).sum().item() / len(label)
        counter += 1

        print(counter, loss.item(), accuracy)

