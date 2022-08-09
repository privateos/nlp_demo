from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sentences = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。'
]
# print(tokenizer, sentences)


#简单编码函数
out = tokenizer.encode(
    text=sentences[0],
    text_pair=sentences[1],

    truncation=True,#句子长度大于max_length时,截断,

    padding='max_length',#一律补pad到max_length长度

    add_special_tokens=True,#是否添加[CLS],[SEP]
    max_length=30,
    return_tensors=None,#返回列表#'tf','pt', 'np'
)

print(out)
print(tokenizer.decode(out))


#增强编码函数
#out = {
# 'input_ids':[0, 20, 11,...],
# 'token_type_ids': [0, 0, 0, 1, 1, 1, 1, ..., 0, 0, 0],
# 'special_tokens_mask': [1, 0, 0, ]#特殊符号位置为1，其他为0
# 'attention_mask_pad':[1,1,1,1,1, 0, 0, 0]#pad的位置为0，其他位置为1
# 'length':30
# }
out = tokenizer.encode_plus(
    text=sentences[0],
    text_pair=sentences[1],

    truncation=True,#当句子长度大于max_length时截断

    padding='max_length',#一律补0到max_length长度
    max_length=30,
    add_special_tokens=True,

    return_tensors=None,#'tf', 'pt', 'np',默认list

    return_token_type_ids=True,#返回token_type_ids

    return_attention_mask=True,#返回attention_mask

    return_special_tokens_mask=True,#返回special_tokens_mask特殊符号标识
)


#批量编码句子/批量成对编码
out = tokenizer.batch_encode_plus(
    # batch_text_or_text_pairs = [sentences[0], sentences[1]],
    batch_text_text_pairs = [(sentences[0], sentences[1]), (sentences[2], sentences[3])],
    add_special_tokens=True,

    truncation=True,#

    padding='max_length',
    max_length=15,

    return_tensors=None,#tf, pt, np

    return_token_type_ids=True,

    return_attention_mask=True,

    return_special_tokens_mask=True,
)
for k, v in out.items():
    print(k, ':', v)
print(tokenizer.decode(out['input_ids'][0]), tokenizer.decode(out['input_ids'][1]))



#获取字典
zidian = tokenizer.get_vocab()#zidian is dict

tokenizer.add_tokens(new_tokens=['月光','希望'])#添加新词到tokenizer中

tokenizer.add_special_tokens({'eos_token':'[EOS]'})#添加新符号

