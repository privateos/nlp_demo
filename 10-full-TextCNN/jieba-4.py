import jieba
# x = '我是大肥猪'
# x = '我是死肥猪,特别肥的死肥猪'
x = '中华女子学院：本科层次仅1专业招男生'

splited_x = jieba.lcut(x)#list(jieba.cut(x))
print(splited_x)
# def my_range(start, end, step=1):
#     current = start
#     while current < end:
#         yield current
#         current += step

# for e in generator:
#     print(e)
# # for i in range(0, 100, 1):
# #     print(i)
# for i in my_range(0, 100, step=1):
#     print(i)
# L = [1,2,3,4,5]
# L1 = [e**2, for e in L]
# L2 = [math.log(e) for e in L1]
# L2 = [math.log(e**2) for e in L]
# def f1(e):
#     return e**2
# import math
# def f2(e):
#     return math.log(e)

# def L_L1(L):
#     for e in L:
#         yield f1(e)

# def L1_L2(L1):
#     for e in L1:
#         yield f2(e)

# result = list(L1_L2(L_L1(L)))
# print(result)
#RNN
#x = (batch_size, pad_size, embedding_dim)
# h_t_1
# for t in range(pad_size):
#     x_t = x[:, t, :]#x_t.shape = (batch_size, embedding_dim)
#     h_t, c_t = lstm(x_t, h_t_1, c_t_1)

