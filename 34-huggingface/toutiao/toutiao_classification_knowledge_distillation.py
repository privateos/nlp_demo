from itertools import count
from re import L
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import os
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


print('teacher')
lr = 1e-4
batch_size = 256
shuffle = True
epochs = 0

dataloader = data.DataLoader(toutiao_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
teacher_model = ToutiaoClassification(hidden_size=768, num_classes=17)
optimizer = optim.AdamW(teacher_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

teacher_model.train()
for epoch in range(epochs):
    counter = 0
    for input_ids, attention_mask, token_type_ids, label in dataloader:
        prediciton = teacher_model(input_ids=input_ids,
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



current_path = os.path.dirname(__file__)
save_teacher_path = os.path.join(current_path, 'teacher.pt')
# torch.save(teacher_model.state_dict(), save_teacher_path)
# exit()
print('load teacher state dict')
teacher_state_dict = torch.load(save_teacher_path)
teacher_model.load_state_dict(teacher_state_dict)

class StudentModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_classes):
        super(StudentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            #(batch_size, hidden_size)
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            #(batch_size, hidden_size)
            nn.Linear(hidden_size, n_classes),
            #(batch_size, n_classes)
        )
    
    def forward(self, input_ids):
        #input_ids.shape = (batch_size, seq_len)

        #embedding.shape = (batch_size, seq_len, hidden_size)
        embedding = self.embedding(input_ids)

        #rnn.shape = (batch_size, seq_len, hidden_size)
        rnn, _ = self.rnn(embedding)

        #last.shape = (batch_size, hidden_size)
        last = rnn[:, -1, :]
        
        #.shape = (batch_size, n_classes)
        return self.fc(last)

print('student')
epochs = 1
hard_loss = nn.CrossEntropyLoss()
soft_loss = nn.KLDivLoss(reduction='batchmean')
vocab = bert_tokenizer.get_vocab()#dict
student_model = StudentModel(vocab_size=len(vocab), hidden_size=256, n_classes=17)
student_optimizer = optim.AdamW(student_model.parameters(), lr=lr)
save_student_path = os.path.join(current_path, 'student.pt')
if os.path.exists(save_student_path):
    print('load student state dict')
    student_state_dict = torch.load(save_student_path)
    student_model.load_state_dict(student_state_dict)
temp = 1.0
alpha = 0.95
student_model.train()
teacher_model.eval()
for epoch in range(epochs):
    counter = 0
    for input_ids, attention_mask, token_type_ids, label in dataloader:
        with torch.no_grad():
            teacher_prediciton = teacher_model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        
        student_prediction = student_model(input_ids)

        student_hard_loss = hard_loss(student_prediction, label)
        student_soft_loss = soft_loss(
            F.log_softmax(student_prediction/temp, dim=-1),
            F.softmax(teacher_prediciton/temp, dim=-1))
        
        student_loss = alpha*student_soft_loss + (1.0 - alpha)*student_hard_loss

        student_optimizer.zero_grad()
        student_loss.backward()
        # print(student_model.embedding.weight.grad);input()
        student_optimizer.step()
        counter += 1

        student_prediction = student_prediction.argmax(dim=1)
        accuracy = (student_prediction == label).sum().item() / len(label)


        print(counter, student_loss.item(), accuracy)
        if counter%20 == 0:
            torch.save(student_model.state_dict(), save_student_path)

torch.save(student_model.state_dict(), save_student_path)