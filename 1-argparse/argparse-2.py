#在终端中打入一下命令
#python argparse-2.py --a 3 --b 4


import argparse


parser = argparse.ArgumentParser()


#首先定义两个变量a,b以便于python可以识别(两个都是int型)
parser.add_argument('--a', type=int, required=True)#required=True表示在终端中这个数据必须给出
parser.add_argument('--b', type=int, required=True)


#解析出这两个参数
args = parser.parse_args()

#打印这两个数以及它们的和
print('a='+str(args.a))
print('b='+str(args.b))
print('a+b='+str(args.a + args.b))
