tokenizer_lambda = lambda x : [y for y in x]

def tokenizer(x):
    # result = []
    # for y in x:
    #     result.append(y)
    # return result
    return [y for y in x]
# print(tokenizer('我是大肥猪'))
print(tokenizer_lambda('我是大肥猪'))
